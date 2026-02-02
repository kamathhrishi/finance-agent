"""
Centralized authentication utilities for the StrataLens API.

Supports both Clerk (primary) and legacy JWT authentication.
"""
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncpg
import jwt

from .jwt_config import verify_clerk_token, decode_access_token, is_clerk_token
from config import settings

logger = logging.getLogger(__name__)

# Security setup - auto_error=False when auth is disabled to allow bypass
security = HTTPBearer(auto_error=not settings.APPLICATION.AUTH_DISABLED)

# Import database utilities
from db.db_utils import get_db

# Dev user for auth bypass
DEV_USER = {
    "id": "00000000-0000-0000-0000-000000000001",
    "username": "dev_user",
    "email": "dev@localhost",
    "full_name": "Development User",
    "is_admin": True
}


async def get_or_create_user_from_clerk(
    clerk_user_id: str,
    payload: Dict[str, Any],
    db: asyncpg.Connection
) -> Dict[str, Any]:
    """
    Get or create a local user record from Clerk user data.

    This function:
    1. First looks up user by clerk_user_id
    2. If not found, creates a new user with Clerk data
    3. Creates default user preferences

    Args:
        clerk_user_id: The Clerk user ID (sub claim)
        payload: The verified Clerk token payload
        db: Database connection

    Returns:
        Dict with user data
    """
    # Extract user info from Clerk payload
    # Clerk JWT may not include email - it's in the user object, not always in the token
    email = payload.get("email") or payload.get("primary_email_address")

    # If no email in token, use a placeholder based on clerk_user_id
    # This allows user creation to succeed; email can be updated later via webhook
    if not email:
        email = f"{clerk_user_id}@clerk.user"
        logger.info(f"🔐 No email in Clerk token, using placeholder: {email}")

    full_name = payload.get("name") or payload.get("full_name") or ""
    first_name = payload.get("first_name", "")
    last_name = payload.get("last_name", "")

    if not full_name and (first_name or last_name):
        full_name = f"{first_name} {last_name}".strip()

    # Try to find user by clerk_user_id
    user = await db.fetchrow(
        """SELECT id, username, email, full_name, is_active, is_approved, is_admin
           FROM users WHERE clerk_user_id = $1""",
        clerk_user_id
    )

    if user:
        logger.info(f"🔐 Found existing user by clerk_user_id: {user['username']}")
        return {
            "id": str(user['id']),
            "username": user['username'],
            "email": user['email'],
            "full_name": user['full_name'],
            "is_admin": user['is_admin'],
            "clerk_user_id": clerk_user_id
        }

    # Try to find user by email (for migration scenarios)
    if email:
        user = await db.fetchrow(
            """SELECT id, username, email, full_name, is_active, is_approved, is_admin, clerk_user_id
               FROM users WHERE email = $1""",
            email
        )

        if user:
            # Link existing user to Clerk
            if not user['clerk_user_id']:
                await db.execute(
                    "UPDATE users SET clerk_user_id = $1 WHERE id = $2",
                    clerk_user_id, user['id']
                )
                logger.info(f"🔗 Linked existing user {user['username']} to Clerk ID: {clerk_user_id}")

            return {
                "id": str(user['id']),
                "username": user['username'],
                "email": user['email'],
                "full_name": user['full_name'],
                "is_admin": user['is_admin'],
                "clerk_user_id": clerk_user_id
            }

    # Create new user from Clerk data
    username = email.split("@")[0] if email else f"user_{clerk_user_id[:8]}"

    # Ensure username is unique
    base_username = username
    counter = 1
    while True:
        existing = await db.fetchrow("SELECT id FROM users WHERE username = $1", username)
        if not existing:
            break
        username = f"{base_username}{counter}"
        counter += 1

    # Insert new user
    user_id = await db.fetchval(
        """INSERT INTO users (
            clerk_user_id, username, email, full_name, first_name, last_name,
            is_active, is_approved, onboarded_via_invitation
        ) VALUES ($1, $2, $3, $4, $5, $6, TRUE, TRUE, FALSE)
        RETURNING id""",
        clerk_user_id, username, email, full_name or username, first_name, last_name
    )

    # Create default preferences
    await db.execute("INSERT INTO user_preferences (user_id) VALUES ($1)", user_id)

    logger.info(f"👤 Created new user from Clerk: {username} (clerk_id: {clerk_user_id})")

    return {
        "id": str(user_id),
        "username": username,
        "email": email,
        "full_name": full_name or username,
        "is_admin": False,
        "clerk_user_id": clerk_user_id
    }


async def authenticate_user_by_token(
    token: str,
    db: asyncpg.Connection,
    auth_type: str = "MAIN"
) -> Dict[str, Any]:
    """
    Common authentication logic for all endpoints.

    Supports both Clerk (RS256) and legacy (HS256) tokens.
    """
    try:
        logger.info(f"🔐 {auth_type} AUTH: Verifying token: {token[:20]}...")

        # Determine token type and verify
        if is_clerk_token(token):
            # Clerk token - verify with JWKS
            payload = verify_clerk_token(token)
            clerk_user_id = payload.get("sub")

            if not clerk_user_id:
                logger.error(f"🔐 {auth_type} AUTH: Clerk token missing 'sub' field")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )

            logger.info(f"🔐 {auth_type} AUTH: Clerk token valid for: {clerk_user_id}")

            # Get or create user from Clerk data
            user = await get_or_create_user_from_clerk(clerk_user_id, payload, db)
            return user

        else:
            # Legacy token - verify with shared secret
            payload = decode_access_token(token)
            user_id: str = payload.get("sub")

            if user_id is None:
                logger.error(f"🔐 {auth_type} AUTH: Token payload missing 'sub' field")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )

            logger.info(f"🔐 {auth_type} AUTH: Legacy token valid for user_id: {user_id}")

            # Verify user exists and is active
            user = await db.fetchrow(
                """SELECT id, username, email, full_name, is_active, is_approved, is_admin
                   FROM users WHERE id = $1""",
                uuid.UUID(user_id)
            )

            if not user:
                logger.error(f"🔐 {auth_type} AUTH: User not found in database: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )

            if not user['is_active'] or not user['is_approved']:
                logger.error(f"🔐 {auth_type} AUTH: User not active or approved: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account not active or not approved"
                )

            logger.info(f"🔐 {auth_type} AUTH: User authenticated successfully: {user['username']}")
            return {
                "id": str(user['id']),
                "username": user['username'],
                "email": user['email'],
                "full_name": user['full_name'],
                "is_admin": user['is_admin']
            }

    except jwt.ExpiredSignatureError:
        logger.error(f"🔐 {auth_type} AUTH: Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"🔐 {auth_type} AUTH: JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔐 {auth_type} AUTH: Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: asyncpg.Connection = Depends(get_db)
) -> Dict[str, Any]:
    """Get current authenticated user - standard dependency for all endpoints."""
    # Check if auth is disabled for local development
    if settings.APPLICATION.AUTH_DISABLED:
        logger.warning("⚠️ AUTH DISABLED - returning dev user (dev mode only)")
        return DEV_USER

    # Credentials will be None if no token provided and auto_error=False
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    return await authenticate_user_by_token(token, db, "MAIN")


async def get_current_user_for_stream(
    request: Request,
    db: asyncpg.Connection = Depends(get_db)
) -> Dict[str, Any]:
    """Get current user for streaming endpoints (supports token in header or query)."""
    # Check if auth is disabled for local development
    if settings.APPLICATION.AUTH_DISABLED:
        logger.warning("⚠️ AUTH DISABLED - returning dev user for stream (dev mode only)")
        return DEV_USER

    # Get token from header or query
    token = None
    try:
        # Try to get from Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            # Try to get from query parameter
            token = request.query_params.get("token")
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication token required. Provide token in Authorization header or as query parameter.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await authenticate_user_by_token(token, db, "STREAM")


async def get_optional_user(
    request: Request,
    db: asyncpg.Connection = Depends(get_db)
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None if not - for optional auth endpoints."""
    # Check if auth is disabled for local development
    if settings.APPLICATION.AUTH_DISABLED:
        return DEV_USER

    try:
        # Try to get from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            return await authenticate_user_by_token(token, db, "OPTIONAL")
    except Exception as e:
        logger.info(f"🔓 Optional auth failed (this is ok for anonymous requests): {e}")

    return None


async def authenticate_user_by_id(user_id: str, db_pool) -> Dict[str, Any]:
    """Authenticate user by ID for WebSocket connections."""
    try:
        async with db_pool.acquire() as db:
            # Verify user exists and is active
            user = await db.fetchrow(
                """SELECT id, username, email, full_name, is_active, is_approved, is_admin
                   FROM users WHERE id = $1""",
                uuid.UUID(user_id)
            )

        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        if not user['is_active'] or not user['is_approved']:
            raise HTTPException(
                status_code=403,
                detail="Account not active or not approved"
            )

        logger.info(f"🔐 WebSocket authenticated user: {user['username']}")
        return {
            "id": str(user['id']),
            "username": user['username'],
            "email": user['email'],
            "full_name": user['full_name'],
            "is_admin": user['is_admin']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ WebSocket authentication error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication failed"
        )

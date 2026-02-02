"""
Authentication routes for StrataLens API.

This module now uses Clerk for authentication:
- User login/signup is handled by Clerk frontend components
- JWT tokens are verified using Clerk's JWKS
- Webhooks sync user data from Clerk to local database

Legacy password hashing is kept for backward compatibility (admin account creation).
"""
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException, Request, Depends, status, Header
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
import asyncpg
import uuid
import logging
import hmac
import hashlib
import json

from passlib.context import CryptContext

from config import settings
from .auth_utils import get_current_user, get_or_create_user_from_clerk
from .jwt_config import verify_clerk_token

# Password hashing context (kept for legacy/admin account support)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (legacy - kept for admin account creation)"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash (legacy - kept for backward compatibility)"""
    return pwd_context.verify(plain_password, hashed_password)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import comprehensive logging
from app.utils.logging_utils import log_info, log_error, log_warning

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

# Database dependency - will be set by main app
def get_db_dependency():
    """
    This will be set by the main app to provide the actual database dependency.
    This allows the auth module to work independently while still accessing the shared db_pool.
    """
    raise HTTPException(status_code=503, detail="Database dependency not initialized")

# This will be set by the main app
get_db = get_db_dependency

def set_db_dependency(db_func):
    """Called by main app to set the database dependency function"""
    global get_db
    get_db = db_func


# =============================================================================
# CLERK WEBHOOK MODELS
# =============================================================================

class ClerkWebhookData(BaseModel):
    """Model for Clerk webhook event data."""
    id: str
    email_addresses: Optional[list] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    primary_email_address_id: Optional[str] = None
    username: Optional[str] = None
    image_url: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class ClerkWebhookEvent(BaseModel):
    """Model for Clerk webhook events."""
    type: str
    data: Dict[str, Any]


# =============================================================================
# WEBHOOK VERIFICATION
# =============================================================================

def verify_clerk_webhook_signature(
    payload: bytes,
    svix_id: str,
    svix_timestamp: str,
    svix_signature: str,
    webhook_secret: str
) -> bool:
    """
    Verify the Clerk webhook signature using Svix.

    Clerk uses Svix for webhook delivery, which signs payloads with HMAC-SHA256.
    """
    if not webhook_secret:
        logger.error("❌ CLERK_WEBHOOK_SECRET not configured")
        return False

    try:
        # Remove 'whsec_' prefix if present
        if webhook_secret.startswith("whsec_"):
            secret = webhook_secret[6:]
        else:
            secret = webhook_secret

        # Decode the base64 secret
        import base64
        secret_bytes = base64.b64decode(secret)

        # Create the signed content
        signed_content = f"{svix_id}.{svix_timestamp}.{payload.decode('utf-8')}"

        # Compute the expected signature
        expected_signature = hmac.new(
            secret_bytes,
            signed_content.encode('utf-8'),
            hashlib.sha256
        ).digest()
        expected_signature_b64 = base64.b64encode(expected_signature).decode('utf-8')

        # The signature header may contain multiple signatures
        signatures = svix_signature.split(" ")
        for sig in signatures:
            if sig.startswith("v1,"):
                sig_value = sig[3:]
                if hmac.compare_digest(sig_value, expected_signature_b64):
                    return True

        logger.warning("⚠️ Webhook signature verification failed")
        return False

    except Exception as e:
        logger.error(f"❌ Error verifying webhook signature: {e}")
        return False


# =============================================================================
# CLERK WEBHOOK ENDPOINT
# =============================================================================

@router.post("/clerk/webhook")
async def clerk_webhook(
    request: Request,
    svix_id: Optional[str] = Header(None, alias="svix-id"),
    svix_timestamp: Optional[str] = Header(None, alias="svix-timestamp"),
    svix_signature: Optional[str] = Header(None, alias="svix-signature")
):
    """
    Handle Clerk webhook events.

    Events handled:
    - user.created: Create local user record
    - user.updated: Update local user record
    - user.deleted: Deactivate local user
    """
    # Get raw body for signature verification
    body = await request.body()

    # Verify webhook signature in production
    if settings.ENVIRONMENT.is_production:
        if not all([svix_id, svix_timestamp, svix_signature]):
            logger.warning("⚠️ Missing Svix headers in webhook request")
            raise HTTPException(status_code=400, detail="Missing webhook headers")

        if not verify_clerk_webhook_signature(
            body, svix_id, svix_timestamp, svix_signature,
            settings.CLERK.WEBHOOK_SECRET
        ):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Parse the event
    try:
        event_data = json.loads(body)
        event_type = event_data.get("type")
        data = event_data.get("data", {})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info(f"📨 Received Clerk webhook: {event_type}")

    # Handle different event types
    async for db in get_db():
        try:
            if event_type == "user.created":
                await handle_user_created(data, db)
            elif event_type == "user.updated":
                await handle_user_updated(data, db)
            elif event_type == "user.deleted":
                await handle_user_deleted(data, db)
            else:
                logger.info(f"ℹ️ Unhandled webhook event type: {event_type}")

            return {"status": "ok", "event": event_type}

        except Exception as e:
            logger.error(f"❌ Error processing webhook: {e}")
            raise HTTPException(status_code=500, detail="Webhook processing failed")


async def handle_user_created(data: Dict[str, Any], db: asyncpg.Connection):
    """Handle user.created webhook event."""
    clerk_user_id = data.get("id")
    if not clerk_user_id:
        logger.error("❌ user.created webhook missing user ID")
        return

    # Extract email from email_addresses array
    email = None
    email_addresses = data.get("email_addresses", [])
    primary_email_id = data.get("primary_email_address_id")

    for email_addr in email_addresses:
        if email_addr.get("id") == primary_email_id:
            email = email_addr.get("email_address")
            break

    if not email and email_addresses:
        email = email_addresses[0].get("email_address")

    # Fallback to placeholder if no email found
    if not email:
        email = f"{clerk_user_id}@clerk.user"
        logger.info(f"🔐 No email in Clerk webhook, using placeholder: {email}")

    first_name = data.get("first_name", "")
    last_name = data.get("last_name", "")
    full_name = f"{first_name} {last_name}".strip() if (first_name or last_name) else email

    # Check if user already exists
    existing = await db.fetchrow(
        "SELECT id FROM users WHERE clerk_user_id = $1 OR email = $2",
        clerk_user_id, email
    )

    if existing:
        # Update existing user with clerk_user_id
        await db.execute(
            "UPDATE users SET clerk_user_id = $1 WHERE id = $2",
            clerk_user_id, existing['id']
        )
        log_info(f"🔗 Linked existing user to Clerk: {email}")
        return

    # Create username from email
    username = email.split("@")[0] if email else f"user_{clerk_user_id[:8]}"

    # Ensure unique username
    base_username = username
    counter = 1
    while await db.fetchrow("SELECT id FROM users WHERE username = $1", username):
        username = f"{base_username}{counter}"
        counter += 1

    # Create new user
    user_id = await db.fetchval(
        """INSERT INTO users (
            clerk_user_id, username, email, full_name, first_name, last_name,
            is_active, is_approved, onboarded_via_invitation
        ) VALUES ($1, $2, $3, $4, $5, $6, TRUE, TRUE, FALSE)
        RETURNING id""",
        clerk_user_id, username, email, full_name, first_name, last_name
    )

    # Create default preferences
    await db.execute("INSERT INTO user_preferences (user_id) VALUES ($1)", user_id)

    log_info(f"👤 Created user from Clerk webhook: {username} ({email})")


async def handle_user_updated(data: Dict[str, Any], db: asyncpg.Connection):
    """Handle user.updated webhook event."""
    clerk_user_id = data.get("id")
    if not clerk_user_id:
        logger.error("❌ user.updated webhook missing user ID")
        return

    # Find user by clerk_user_id
    user = await db.fetchrow(
        "SELECT id FROM users WHERE clerk_user_id = $1",
        clerk_user_id
    )

    if not user:
        logger.warning(f"⚠️ User not found for update: {clerk_user_id}")
        # Try to create the user instead
        await handle_user_created(data, db)
        return

    # Extract updated fields
    email = None
    email_addresses = data.get("email_addresses", [])
    primary_email_id = data.get("primary_email_address_id")

    for email_addr in email_addresses:
        if email_addr.get("id") == primary_email_id:
            email = email_addr.get("email_address")
            break

    first_name = data.get("first_name", "")
    last_name = data.get("last_name", "")
    full_name = f"{first_name} {last_name}".strip()

    # Update user
    await db.execute(
        """UPDATE users SET
            email = COALESCE($1, email),
            full_name = CASE WHEN $2 != '' THEN $2 ELSE full_name END,
            first_name = COALESCE($3, first_name),
            last_name = COALESCE($4, last_name)
        WHERE clerk_user_id = $5""",
        email, full_name, first_name, last_name, clerk_user_id
    )

    log_info(f"✏️ Updated user from Clerk webhook: {clerk_user_id}")


async def handle_user_deleted(data: Dict[str, Any], db: asyncpg.Connection):
    """Handle user.deleted webhook event."""
    clerk_user_id = data.get("id")
    if not clerk_user_id:
        logger.error("❌ user.deleted webhook missing user ID")
        return

    # Deactivate user (soft delete)
    result = await db.execute(
        "UPDATE users SET is_active = FALSE WHERE clerk_user_id = $1",
        clerk_user_id
    )

    log_info(f"🗑️ Deactivated user from Clerk webhook: {clerk_user_id}")


# =============================================================================
# TOKEN VALIDATION ENDPOINT
# =============================================================================

@router.get("/validate")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """
    Validates the user's token (works with both Clerk and legacy tokens).
    If the token is valid, returns success with user info.
    """
    return {
        "status": "ok",
        "message": "Token is valid",
        "user_id": current_user["id"],
        "user": {
            "id": current_user["id"],
            "username": current_user.get("username"),
            "email": current_user.get("email"),
            "full_name": current_user.get("full_name"),
            "is_admin": current_user.get("is_admin", False)
        }
    }


# =============================================================================
# USER INFO ENDPOINT
# =============================================================================

@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user's information."""
    return {
        "id": current_user["id"],
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "full_name": current_user.get("full_name"),
        "is_admin": current_user.get("is_admin", False)
    }


# =============================================================================
# CORS OPTIONS HANDLERS
# =============================================================================

@router.options("/validate")
async def options_validate():
    return JSONResponse(content={"message": "OK"})


@router.options("/me")
async def options_me():
    return JSONResponse(content={"message": "OK"})


@router.options("/clerk/webhook")
async def options_clerk_webhook():
    return JSONResponse(content={"message": "OK"})


# =============================================================================
# CLERK CONFIGURATION ENDPOINT (for frontend)
# =============================================================================

@router.get("/clerk/config")
async def get_clerk_config():
    """
    Get Clerk configuration for frontend initialization.

    Returns the publishable key (safe to expose) for Clerk.js initialization.
    """
    if not settings.CLERK.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Clerk authentication is not configured"
        )

    return {
        "publishableKey": settings.CLERK.PUBLISHABLE_KEY,
        "signInUrl": "/sign-in",
        "signUpUrl": "/sign-up",
        "afterSignInUrl": "/",
        "afterSignUpUrl": "/"
    }

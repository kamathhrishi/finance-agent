"""
Centralized JWT configuration for StrataLens.

This module handles both:
1. Clerk JWT verification (primary - uses JWKS)
2. Legacy JWT operations (deprecated - for backwards compatibility)
"""
import os
import time
import logging
import httpx
from typing import Optional, Dict, Any, Tuple

import jwt
from jwt import PyJWKClient, PyJWKClientError

from config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# LEGACY JWT CONFIGURATION (Deprecated - kept for backwards compatibility)
# =============================================================================

_SECRET_KEY = None


def get_secret_key():
    """Get the SECRET_KEY, generating one if needed (LEGACY - deprecated)"""
    global _SECRET_KEY
    if _SECRET_KEY is None:
        import secrets
        _SECRET_KEY = os.getenv("JWT_SECRET_KEY")
        if not _SECRET_KEY:
            _SECRET_KEY = secrets.token_urlsafe(32)
            logger.warning("⚠️  Generated random JWT secret key (will change on restart). Set JWT_SECRET_KEY environment variable for production!")
    return _SECRET_KEY


# Legacy constants (deprecated)
SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT access token (LEGACY - deprecated)"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")


# =============================================================================
# CLERK JWT VERIFICATION (Primary)
# =============================================================================

class ClerkJWKSClient:
    """
    Client for fetching and caching Clerk's JWKS (JSON Web Key Set).

    Clerk tokens are signed with RS256 and verified using public keys
    fetched from Clerk's JWKS endpoint.
    """

    def __init__(self):
        self._jwks_clients: Dict[str, Tuple[PyJWKClient, float]] = {}
        self._cache_ttl = settings.CLERK.JWKS_CACHE_TTL_SECONDS

    def _get_jwks_url_from_issuer(self, issuer: str) -> str:
        """Construct JWKS URL from the token issuer."""
        # Clerk issuer format: https://<clerk-id>.clerk.accounts.dev or https://clerk.<domain>
        # JWKS endpoint: <issuer>/.well-known/jwks.json
        return f"{issuer.rstrip('/')}/.well-known/jwks.json"

    def get_signing_key(self, token: str) -> Any:
        """
        Get the signing key for a Clerk JWT token.

        This method:
        1. Decodes the token header to get the issuer
        2. Fetches/caches the JWKS from that issuer
        3. Returns the signing key matching the token's kid
        """
        try:
            # Decode header without verification to get issuer
            unverified_header = jwt.get_unverified_header(token)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})

            issuer = unverified_payload.get("iss")
            if not issuer:
                raise jwt.InvalidTokenError("Token missing issuer (iss) claim")

            jwks_url = self._get_jwks_url_from_issuer(issuer)

            # Check cache
            now = time.time()
            if jwks_url in self._jwks_clients:
                client, cached_at = self._jwks_clients[jwks_url]
                if now - cached_at < self._cache_ttl:
                    return client.get_signing_key_from_jwt(token)

            # Create new client and cache it
            logger.info(f"🔑 Fetching JWKS from: {jwks_url}")
            client = PyJWKClient(jwks_url)
            self._jwks_clients[jwks_url] = (client, now)

            return client.get_signing_key_from_jwt(token)

        except PyJWKClientError as e:
            logger.error(f"❌ JWKS client error: {e}")
            raise jwt.InvalidTokenError(f"Failed to fetch signing key: {e}")
        except Exception as e:
            logger.error(f"❌ Error getting signing key: {e}")
            raise


# Global JWKS client instance
_clerk_jwks_client = None


def get_clerk_jwks_client() -> ClerkJWKSClient:
    """Get or create the global JWKS client."""
    global _clerk_jwks_client
    if _clerk_jwks_client is None:
        _clerk_jwks_client = ClerkJWKSClient()
    return _clerk_jwks_client


def verify_clerk_token(token: str) -> Dict[str, Any]:
    """
    Verify a Clerk JWT token and return the payload.

    Args:
        token: The JWT token from Clerk

    Returns:
        Dict containing the verified token payload with claims like:
        - sub: Clerk user ID (e.g., "user_2abc123")
        - email: User's email
        - name: User's full name
        - iss: Token issuer
        - exp: Expiration timestamp
        - iat: Issued at timestamp

    Raises:
        jwt.ExpiredSignatureError: If token has expired
        jwt.InvalidTokenError: If token is invalid
    """
    if not settings.CLERK.is_configured:
        logger.error("❌ Clerk is not configured. Set CLERK_SECRET_KEY and CLERK_PUBLISHABLE_KEY.")
        raise jwt.InvalidTokenError("Clerk authentication not configured")

    try:
        # Get signing key from JWKS
        jwks_client = get_clerk_jwks_client()
        signing_key = jwks_client.get_signing_key(token)

        # Verify and decode the token
        # Add leeway to handle clock skew between server and Clerk
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": False,  # Clerk doesn't always set audience
            },
            leeway=30  # Allow 30 seconds of clock skew for iat/exp validation
        )

        logger.debug(f"✅ Clerk token verified for user: {payload.get('sub')}")
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("⚠️ Clerk token has expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.warning(f"⚠️ Invalid Clerk token: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error verifying Clerk token: {e}")
        raise jwt.InvalidTokenError(f"Token verification failed: {e}")


def is_clerk_token(token: str) -> bool:
    """
    Check if a token is a Clerk token (RS256 signed).

    Clerk tokens use RS256, while legacy tokens use HS256.
    """
    try:
        header = jwt.get_unverified_header(token)
        return header.get("alg") == "RS256"
    except Exception:
        return False


logger.info("🔐 JWT Config initialized (Clerk + Legacy support)")

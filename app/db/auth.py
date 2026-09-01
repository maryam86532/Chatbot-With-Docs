import hmac

from app.core.config import config


def get_user_id_from_token(token: str) -> str | None:
    """
    Validates the provided Bearer token against the CHAT_API_KEY from env.
    Returns a fixed user identifier if the key matches, else None.

    This is a simple shared-secret API key check. For per-user auth, replace
    this with JWT / session / database-backed logic.
    """
    expected = config.CHAT_API_KEY
    if not expected:
        return None
    # Protected against timing attacks
    if hmac.compare_digest(str(token), str(expected)):
        return "local-user"
    return None

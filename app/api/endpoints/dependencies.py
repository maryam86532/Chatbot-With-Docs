from fastapi import Request, HTTPException, status

def get_current_user_id(request: Request) -> str:
    """
    Dependency to get the user ID from the request state.
    This assumes that the AccessTokenAuthMiddleware has already run
    and set request.state.user_id.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        # This case should ideally not be reached if the middleware is applied correctly
        # to all protected routes.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

import base64
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.types import ASGIApp
from starlette.concurrency import run_in_threadpool

from app.core.config import config
from app.db.auth import get_user_id_from_token

DOCS_USERNAME = config.DOCS_USERNAME
DOCS_PASSWORD = config.DOCS_PASSWORD

PROTECTED_PATHS = {"/docs", "/redoc", "/openapi.json"}

class DocsAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PROTECTED_PATHS:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Basic "):
                return self._unauthorized_response()

            try:
                encoded_credentials = auth_header.split(" ")[1]
                decoded = base64.b64decode(encoded_credentials).decode("utf-8")
                username, password = decoded.split(":", 1)

                if username != DOCS_USERNAME or password != DOCS_PASSWORD:
                    return self._unauthorized_response()

            except Exception:
                return self._unauthorized_response()

        return await call_next(request)

    def _unauthorized_response(self) -> Response:
        return PlainTextResponse(
            "Unauthorized access to docs or OpenAPI schema",
            status_code=HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Basic"},
        )

# AccessTokenAuthMiddleware
class AccessTokenAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    - Requires Authorization header with Bearer token
    - Validates token against the database
    - Blocks unauthorized requests with 401
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:

        if self.is_public_route(request):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid authorization token"})

        token = auth_header.removeprefix("Bearer ").strip()
        
        user_id = await run_in_threadpool(get_user_id_from_token, token)

        if not user_id:
            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})

        request.state.user_id = user_id

        return await call_next(request)

    def is_public_route(self, request: StarletteRequest) -> bool:
        path = request.url.path
        return (
            path == "/docs"
            or path == "/openapi.json"
            or path == "/redoc"
            or path == "/"
            or path.startswith("/server_info")
            or path.startswith("/static")
        )

    def is_json_request(self, request: StarletteRequest) -> bool:
        return request.headers.get("content-type", "").startswith("application/json")
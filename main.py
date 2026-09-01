from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.middleware.auth import AccessTokenAuthMiddleware, DocsAuthMiddleware
from app.middleware.cors import add_cors_middleware
from app.core.openapi import custom_openapi
from app.core.config import config
from app.db.database import initialize_database

from app.api.endpoints.routes import (
    chat,
    server_info,
)

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
)

@app.on_event("startup")
async def startup_event():
    initialize_database()

# Add middlewares
add_cors_middleware(app)
app.add_middleware(DocsAuthMiddleware)
app.add_middleware(AccessTokenAuthMiddleware)

# Include routers
app.include_router(server_info.router)
app.include_router(chat.router)

# Serve uploaded images so AI-analysed images can be re-displayed in chat history.
from app.core.config import ensure_data_dir
app.mount(
    "/static/uploads",
    StaticFiles(directory=ensure_data_dir(config.IMAGE_UPLOAD_DIR), check_dir=False),
    name="uploads",
)

# Serve the static HTML UI at the root
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")


@app.get("/", include_in_schema=False)
async def index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)

from fastapi.middleware.cors import CORSMiddleware
from app.core.config import config

def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in config.CORS_ALLOWED_ORIGINS.split(',')],
        allow_credentials=config.CORS_ALLOW_CREDENTIALS,
        allow_methods=[method.strip() for method in config.CORS_ALLOWED_METHODS.split(',')],
        allow_headers=[header.strip() for header in config.CORS_ALLOWED_HEADERS.split(',')],
    )
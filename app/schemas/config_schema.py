from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application details
    APP_TITLE: str = "RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "RAG chatbot powered by FastAPI, LangChain, Google Gemini, FAISS, and SQLite."

    # CORS
    CORS_ALLOWED_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOWED_METHODS: str = "*"
    CORS_ALLOWED_HEADERS: str = "*"

    # Docs Auth
    DOCS_USERNAME: str = "admin"
    DOCS_PASSWORD: str = "admin"

    # Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"

    # Chat API Authentication (simple API key)
    CHAT_API_KEY: str = "change-me"

    # SQLite Database
    SQLITE_DB_PATH: str = "data/chatbot.db"

    # FAISS Vector Store
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # Uploaded image storage
    IMAGE_UPLOAD_DIR: str = "data/images"

    # Logging
    LOGLEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

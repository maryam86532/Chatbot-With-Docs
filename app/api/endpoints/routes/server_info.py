from fastapi import APIRouter
from app.core.config import config
from app.core.vectorstore import _get_vectorstore
from app.db.database import db_cursor

router = APIRouter(
    prefix="/server_info",
    tags=["Server Info"],
)

@router.get("/health", response_model=dict)
async def health_check():
    return {"status": "ok"}

@router.get("/readiness", response_model=dict)
async def readiness_check():
    try:
        with db_cursor() as cur:
            cur.execute("SELECT 1")
        return {"ready": True}
    except Exception:
        return {"ready": False}

@router.get("/liveness", response_model=dict)
async def liveness_check():
    return {"alive": True}

@router.get("/version", response_model=dict)
async def version_info():
    return {"title": config.APP_TITLE, "version": config.APP_VERSION, "description": config.APP_DESCRIPTION}

@router.get("/status", response_model=dict)
async def rag_status():
    """Report whether the RAG knowledge base has been initialized."""
    try:
        vs = _get_vectorstore()
        has_index = vs is not None
        doc_count = vs.index.ntotal if has_index else 0
    except Exception:
        has_index = False
        doc_count = 0
    return {
        "vector_store_initialized": has_index,
        "documents_indexed": doc_count,
        "gemini_configured": bool(config.GEMINI_API_KEY),
    }

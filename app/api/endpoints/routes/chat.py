import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.rag import answer_question, answer_question_with_images, ensure_session
from app.core.vectorstore import (
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    index_document,
    save_image,
)
from app.db.database import db_cursor

router = APIRouter(prefix="/chat", tags=["Chat"])


class NewSessionRequest(BaseModel):
    title: str = Field(default="New chat", max_length=200)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=8000)


class SessionListResponse(BaseModel):
    sessions: list


@router.post("/sessions", response_model=dict)
async def create_session(body: NewSessionRequest):
    session_id = ensure_session(None, title=body.title)
    return {"session_id": session_id, "title": body.title}


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    with db_cursor() as cur:
        cur.execute("SELECT id, title FROM sessions ORDER BY updated_at DESC")
        rows = cur.fetchall()
    return {"sessions": [{"id": r["id"], "title": r["title"]} for r in rows]}


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str):
    with db_cursor() as cur:
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"deleted": session_id}


@router.get("/sessions/{session_id}/messages", response_model=dict)
async def get_messages(session_id: str):
    with db_cursor() as cur:
        cur.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = cur.fetchall()
    return {
        "session_id": session_id,
        "messages": [dict(r) for r in rows],
    }


@router.post("/ask", response_model=dict)
async def ask(request: Request):
    """Answer a question. Accepts either JSON (session_id, message) or
    multipart/form-data (session_id, message, images[]) so uploaded images can be
    analyzed by Gemini vision."""
    content_type = request.headers.get("content-type", "")
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            session_id = (form.get("session_id") or "").strip()
            message = (form.get("message") or "").strip()
            image_files = form.getlist("images")
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            if not message:
                raise HTTPException(status_code=400, detail="message is required")
            session_id = ensure_session(session_id)

            images = []
            for f in image_files:
                ext = os.path.splitext(f.filename or "")[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                data = await f.read()
                path = save_image(data, f.filename or "upload")
                images.append({"path": path, "name": f.filename or "upload"})

            if images:
                result = answer_question_with_images(session_id, message, images)
            else:
                result = answer_question(session_id, message)
        else:
            body = await request.json()
            session_id = ensure_session(body.get("session_id"))
            message = (body.get("message") or "").strip()
            if not message:
                raise HTTPException(status_code=400, detail="message is required")
            result = answer_question(session_id, message)
    except HTTPException:
        raise
    except RuntimeError as e:
        # e.g. GEMINI_API_KEY missing or no docs ingested
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the request: {e}")

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }


@router.post("/documents", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    data = await file.read()
    tmp_path = None
    try:
        if ext in IMAGE_EXTENSIONS:
            # Persist the image so it can be captioned, embedded, and re-analysed later.
            target_path = save_image(data, file.filename)
            summary = index_document(target_path)
        else:
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            summary = index_document(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest document: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {
        "message": f"Document '{file.filename}' ingested successfully.",
        "source": summary["source"],
        "chunks": summary["chunks"],
    }

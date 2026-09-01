import os
import time
import uuid
from typing import Callable, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.core.config import config
from app.core.logger import get_logger
from app.core.vectorstore import get_or_create_vectorstore
from app.db.database import db_cursor

logger = get_logger(__name__)

# In-memory chat history keyed by session id (list of AIMessage/HumanMessage).
_histories: Dict[str, list] = {}


def _get_chat_model() -> ChatGoogleGenerativeAI:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in your .env file.")
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=1024,
        max_retries=6,
    )


def _load_history_from_db(session_id: str) -> None:
    """Populate in-memory history from SQLite so sessions survive restarts."""
    if session_id in _histories:
        return
    history: List = []
    with db_cursor() as cur:
        cur.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        for row in cur.fetchall():
            if row["role"] == "user":
                history.append(HumanMessage(content=row["content"]))
            else:
                history.append(AIMessage(content=row["content"]))
    _histories[session_id] = history


def _session_exists(session_id: str) -> bool:
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        return cur.fetchone() is not None


def ensure_session(session_id: Optional[str], title: str = "New chat") -> str:
    """Return a valid session id, creating a new row if needed."""
    if session_id and _session_exists(session_id):
        return session_id
    new_id = str(uuid.uuid4())
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, title) VALUES (?, ?)",
            (new_id, title),
        )
    _histories[new_id] = []
    return new_id


def _retry_on_transient(fn: Callable, attempts: int = 3, base_delay: float = 1.5):
    """Retry a callable on transient Gemini errors (HTTP 429/500/503) with backoff.

    Quota-exhaustion (429 RESOURCE_EXHAUSTED) is treated specially: it is reported
    immediately with a clear message rather than retried (retries would only burn
    more quota).
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - surface after exhausting retries
            last_exc = e
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                logger.warning("Gemini quota exhausted: %s", msg[:200])
                raise RuntimeError(
                    "The Gemini API has hit its rate limit (quota exhausted). "
                    "Please wait a bit and try again, or upgrade your Gemini plan."
                )
            transient = "429" in msg or "500" in msg or "503" in msg or "UNAVAILABLE" in msg
            if not transient or attempt == attempts - 1:
                raise
            logger.warning("Transient error (attempt %s/%s): %s",
                           attempt + 1, attempts, msg[:120])
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc


def _build_direct_chain(llm=None):
    """Build a plain conversational chain (Gemini answers from general knowledge)."""
    llm = llm or _get_chat_model()
    direct_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful, concise assistant. You do not currently have any "
                "uploaded documents to search, so answer from your general knowledge. "
                "If you are not sure, say so. Be friendly and clear.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return direct_prompt | llm | StrOutputParser()


def _build_chain(capture: Dict[str, list]):
    """Build the RAG LCEL chain. Expects input dict with 'input' and 'chat_history'.

    If no documents have been ingested yet, builds a plain conversational chain
    (Gemini answers from its own knowledge) instead of a retrieval chain.
    """
    llm = _get_chat_model()
    vs = get_or_create_vectorstore(raise_if_missing=False)
    has_docs = vs is not None
    capture["has_docs"] = has_docs

    if not has_docs:
        # Plain conversational answering (no knowledge base yet).
        return _build_direct_chain(llm)

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def capture_retrieve(query: str):
        try:
            docs = _retry_on_transient(lambda: retriever.invoke(query))
        except Exception as e:
            # Transient failure (e.g. Gemini embed 500 / 429). Signal it so the
            # caller can fall back to a general-knowledge answer instead of surfacing
            # a hard error to the user.
            logger.warning("Retrieval failed, will fall back to general answer: %s", str(e)[:200])
            capture["retrieval_failed"] = True
            return []
        capture["documents"] = docs
        return docs

    # 1. Contextualize the latest question using chat history.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question which might reference "
                "context in the history, formulate a standalone question that can be understood "
                "without the chat history. Do NOT answer the question, only reformulate it if "
                "needed, otherwise return it as is.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    def contextualized_question(input_: dict):
        if input_.get("chat_history"):
            return contextualize_q_chain
        return input_["input"]

    # 2. Retrieve context and build the generation chain.
    contextual_retriever = RunnableLambda(contextualized_question) | RunnableLambda(capture_retrieve)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with access to a knowledge base. Answer the "
                "user's question based ONLY on the provided context. If the context does not "
                "contain the answer, say you don't know rather than guessing. Be concise.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(context=contextual_retriever)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_question(session_id: str, question: str) -> dict:
    """Run RAG with memory for the given session and persist the result."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in your .env file.")

    _load_history_from_db(session_id)
    history = _histories.setdefault(session_id, [])
    capture = {}
    rag_chain = _build_chain(capture)

    answer = _retry_on_transient(
        lambda: rag_chain.invoke({"input": question, "chat_history": history}),
        attempts=5,
    )

    if capture.get("retrieval_failed"):
        # Retrieval could not fetch context (transient embedding error). Fall back
        # to a general-knowledge answer so the user still gets a useful response.
        logger.warning("Falling back to general-knowledge answer for session %s", session_id)
        direct_chain = _build_direct_chain()
        answer = _retry_on_transient(
            lambda: direct_chain.invoke({"input": question, "chat_history": history}),
            attempts=3,
        )
        capture["documents"] = []

    sources = _collect_sources(capture.get("documents", []))

    # Persist both turns to SQLite.
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
            (session_id, question),
        )
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
            (session_id, answer),
        )
        cur.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,),
        )

    # Update in-memory history for the next turn.
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    return {"answer": answer, "sources": sources}


def _collect_sources(documents) -> list:
    sources = []
    seen = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def _image_to_data_url(path: str) -> str:
    import base64 as _b64
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + _b64.b64encode(f.read()).decode()


def answer_question_with_images(session_id: str, question: str, images: List[dict]) -> dict:
    """Answer a question that includes uploaded images by sending them to Gemini
    multimodal (vision). RAG context is still added when a knowledge base exists.
    `images` is a list of dicts like {'path': str, 'name': str}.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in your .env file.")

    _load_history_from_db(session_id)
    history = _histories.setdefault(session_id, [])

    # Optionally pull RAG context when documents exist.
    context = ""
    sources = []
    try:
        vs = get_or_create_vectorstore(raise_if_missing=False)
        if vs is not None:
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            try:
                docs = _retry_on_transient(lambda: retriever.invoke(question), attempts=3)
            except Exception as e:
                logger.warning("Retrieval failed for image question, ignoring context: %s", str(e)[:120])
                docs = []
            sources = _collect_sources(docs)
            if docs:
                context = "\n\n".join(d.page_content for d in docs)
    except Exception as e:  # noqa: BLE001 - RAG context is optional here
        logger.warning("Could not gather RAG context for image question: %s", str(e)[:120])

    content = [{"type": "text", "text": question}]
    for img in images:
        try:
            content.append({"type": "image_url", "image_url": _image_to_data_url(img["path"])})
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read image %s: %s", img.get("name"), str(e)[:120])

    system = (
        "You are a helpful assistant. The user has attached image(s) and asked a question. "
        "Analyze the image(s) carefully and answer based on what you see in them. "
    )
    if context:
        system += (
            "\nYou also have access to a knowledge base. Use the following context to help "
            "answer, but prioritize what you observe in the image(s):\n\n"
            f"Context:\n{context}"
        )

    llm = _get_chat_model()
    user_msg = HumanMessage(content=content)
    try:
        answer = _retry_on_transient(lambda: llm.invoke([*history, user_msg]), attempts=5)
        answer = answer.content
        if isinstance(answer, list):
            answer = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in answer
            ).strip()
    except Exception as e:
        raise

    # Persist user message (with image refs) and assistant answer. Image refs use
    # markdown image syntax pointing at the served upload URL so chat history can
    # re-render the image.
    img_lines = []
    for img in images:
        name = img.get("name", "image")
        path = img.get("path", "")
        rel = os.path.basename(path)
        url = f"/static/uploads/{rel}"
        img_lines.append(f"![{name}]({url})")
    user_content = question
    if img_lines:
        user_content += "\n" + "\n".join(img_lines)
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
            (session_id, user_content.strip()),
        )
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
            (session_id, answer),
        )
        cur.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,),
        )

    history.append(HumanMessage(content=user_content.strip()))
    history.append(AIMessage(content=answer))

    return {"answer": answer, "sources": sources}

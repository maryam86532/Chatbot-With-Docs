import hashlib
import os
import uuid
from typing import List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import config, ensure_data_dir
from app.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=config.GEMINI_API_KEY,
        max_retries=6,
    )


def save_image(upload: bytes, filename: str) -> str:
    """Persist an uploaded image under the configured images dir and return its path."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in your .env file.")
    ext = os.path.splitext(filename)[1].lower()
    digest = hashlib.sha1(upload).hexdigest()[:12]
    safe_name = f"{digest}{ext}"
    image_dir = ensure_data_dir(config.IMAGE_UPLOAD_DIR)
    path = os.path.join(image_dir, safe_name)
    with open(path, "wb") as f:
        f.write(upload)
    return path


def _describe_image(file_path: str) -> str:
    """Generate a concise text description of an image using Gemini vision."""
    import base64 as _b64
    ext = os.path.splitext(file_path)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "image/png")
    with open(file_path, "rb") as f:
        data_url = f"data:{mime};base64," + _b64.b64encode(f.read()).decode()

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.2,
        max_output_tokens=300,
        max_retries=6,
    )
    media = HumanMessage(content=[
        {"type": "text", "text": (
            "Describe this image in detail so the description can be stored in a "
            "knowledge base and retrieved later. Include the main subject, any visible "
            "text, colors, and notable details. Keep it under 150 words."
        )},
        {"type": "image_url", "image_url": data_url},
    ])
    try:
        return llm.invoke([media]).content
    except Exception as e:  # noqa: BLE001 - fall back to a generic description
        logger.warning("Could not caption image %s: %s", os.path.basename(file_path), str(e)[:120])
        return f"An uploaded image file named {os.path.basename(file_path)}."


def _load_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in IMAGE_EXTENSIONS:
        # Images are described by Gemini vision so they can be embedded/retrieved.
        description = _describe_image(file_path)
        doc = Document(
            page_content=description,
            metadata={"source": os.path.basename(file_path), "type": "image"},
        )
        return [doc]
    else:  # pragma: no cover
        raise ValueError(f"Unsupported file type '{ext}'.")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", os.path.basename(file_path))
    return docs


def _get_vectorstore(index_path: Optional[str] = None) -> Optional[FAISS]:
    """Load an existing FAISS index from disk, if present."""
    index_path = index_path or config.FAISS_INDEX_PATH
    if os.path.exists(index_path) and os.listdir(index_path):
        try:
            return FAISS.load_local(
                index_path,
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.warning("Could not load existing FAISS index: %s", e)
    return None


def get_or_create_vectorstore(raise_if_missing: bool = True):
    """Return the existing vector store, or None if none has been built yet."""
    vs = _get_vectorstore()
    if vs is None and raise_if_missing:
        raise RuntimeError("No documents have been ingested yet. Please upload a document first.")
    return vs


def index_document(file_path: str) -> dict:
    """
    Load the document at file_path, split it, and add it to the FAISS index.
    Returns a summary dict about what was indexed.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in your .env file.")

    docs = _load_document(file_path)
    if not docs:
        raise ValueError("No readable text found in the uploaded document.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    index_path = ensure_data_dir(config.FAISS_INDEX_PATH)

    vs = _get_vectorstore(index_path)
    if vs is None:
        vs = FAISS.from_documents(chunks, embeddings)
        logger.info("Created new FAISS index with %d chunks.", len(chunks))
    else:
        vs.add_documents(chunks)
        logger.info("Added %d chunks to existing FAISS index.", len(chunks))

    vs.save_local(index_path)
    return {
        "chunks": len(chunks),
        "source": os.path.basename(file_path),
    }
# PERSIST_TEST_MARKER_12345
os.makedirs(config.IMAGE_UPLOAD_DIR, exist_ok=True)

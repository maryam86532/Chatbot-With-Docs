[RAG Chatbot.html](https://github.com/user-attachments/files/31685925/RAG.Chatbot.html)
# RAG Chatbot

A **retrieval-augmented generation (RAG)** chatbot built on FastAPI, LangChain,
Google Gemini, FAISS, and SQLite. Users upload documents **and images**, ask
questions, and the assistant answers **only** from the contents of those
documents (with images analyzed directly by the Gemini vision model).

## Features

- **RAG pipeline**: FAISS vector store + Google Gemini embeddings & chat model.
- **Multi-format ingest**: upload `.txt`, `.md`, `.pdf`, `.docx`, `.png`, `.jpg`,
  `.jpeg`, `.gif`, `.webp`, or `.bmp` files (single or multiple).
- **Vision Q&A**: attach images and ask questions — Gemini analyzes them directly.
  Images can also be uploaded as documents (captioned, embedded, and retrievable).
- **Session chat history**: conversations are stored in SQLite and reloadable from a sidebar.
- **Beautiful hosted UI**: a modern, responsive single-page interface served by FastAPI.
- **Simple API-key auth**: chat/document endpoints require a Bearer token (`Authorization: Bearer <key>`).
- **Health checks**: health / readiness / liveness / version / rag-status endpoints.

## Tech Stack

- **FastAPI** + **Uvicorn**
- **LangChain** / **LangChain-Google-GenAI** (Gemini)
- **FAISS** (vector store)
- **SQLite** (session & message persistence)
- **pypdf** / **python-docx** / **text splitter** for document parsing
- **Gemini vision** for image analysis / captioning

## Getting Started

### Prerequisites

- Python 3.11+ (uses `uv` for the environment)
- A Google Gemini API key: <https://aistudio.google.com/app/apikey>

### Installation

1. **Clone / enter the repo**

2. **Create the environment** (with `uv`):
   ```bash
   uv venv --python 3.11
   uv pip install -r requirements.txt
   ```
   (Or: `python -m venv venv` + `pip install -r requirements.txt`.)

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - `GEMINI_API_KEY` — your Gemini key (required for ingestion & answers).
   - `CHAT_API_KEY` — a secret used as the Bearer token for the API/UI.
   - Optionally `GEMINI_MODEL`, `SQLITE_DB_PATH`, `FAISS_INDEX_PATH`,
     `IMAGE_UPLOAD_DIR` (default `data/images`).

### Run

```bash
uv run uvicorn main:app --reload
# or
.venv\Scripts\python -m uvicorn main:app --reload
```

- UI: <http://127.0.0.1:8000>
- API docs: <http://127.0.0.1:8000/docs> (protected by `DOCS_USERNAME`/`DOCS_PASSWORD`)
- Health: <http://127.0.0.1:8000/server_info/health>

## Usage

1. Open <http://127.0.0.1:8000>.
2. In the sidebar, paste your `CHAT_API_KEY` into the **API Key** field.
3. Upload one or more documents (txt / md / pdf / docx / images). Text documents are
   embedded and stored in the FAISS index; images are saved and (when uploaded as
   documents) captioned for retrieval.
4. Start a new chat and ask questions — answers are grounded in the uploaded documents.
   If you attach images with your question, Gemini analyzes them directly with vision.
5. Existing chats appear in the sidebar and can be reopened.

> Sample knowledge document: `data/sample.md` for a quick first ingest.
> Uploaded images are stored in `data/images/` and served from `/static/uploads/`.

## API Endpoints

All `/chat/*` endpoints require `Authorization: Bearer <CHAT_API_KEY>`.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/server_info/health` | Liveness/health check |
| `GET`  | `/server_info/status` | RAG/vector-store & Gemini status |
| `POST` | `/chat/sessions` | Create a new chat session |
| `GET`  | `/chat/sessions` | List chat sessions |
| `GET`  | `/chat/sessions/{id}/messages` | Get messages for a session |
| `DELETE` | `/chat/sessions/{id}` | Delete a session & its messages |
| `POST` | `/chat/ask` | Ask a question — JSON `{session_id, message}` **or** multipart (`session_id`, `message`, `images[]` for vision) |
| `POST` | `/chat/documents` | Upload a file to ingest (multipart `file`; images and text/PDF/doc supported) |

## Project Structure

```
.
├── app/
│   ├── api/endpoints/
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       └── server_info.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── openapi.py
│   │   ├── rag.py            # RAG chain (LangChain LCEL + Gemini)
│   │   └── vectorstore.py    # Document loading + FAISS indexing
│   ├── db/
│   │   ├── auth.py           # simple API key validation
│   │   └── database.py       # SQLite session/message storage
│   ├── middleware/
│   │   ├── auth.py
│   │   └── cors.py
│   └── schemas/
│       └── config_schema.py
├── static/
│   └── index.html            # hosted chat UI
├── data/
│   ├── sample.md             # sample knowledge document
│   ├── chatbot.db            # SQLite (runtime, gitignored)
│   ├── faiss_index/          # FAISS vectors (runtime, gitignored)
│   └── images/               # uploaded images (runtime, gitignored)
├── .env.example
├── .gitignore
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Notes

- The RAG QA prompt instructs the model to answer **only from retrieved context** and
  to say "I don't know" when the answer isn't present.
- Image questions are answered by the Gemini vision model; RAG context is still added
  when a knowledge base exists, but the model prioritizes what it observes in the images.
- The FAISS index, SQLite database, and uploaded images are stored under `data/` and
  excluded from git.

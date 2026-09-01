# RAG Chatbot Documentation

## Overview

RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative
language model. When a user asks a question, the system first retrieves relevant
chunks of text from a vector database, then feeds those chunks as context to the
language model to produce a grounded answer.

## Components

### Vector Store (FAISS)
FAISS is a library developed by Meta for efficient similarity search and clustering
of dense vectors. It is used here to index and search document embeddings. Supported
document types are plain text (.txt), Markdown (.md), PDF (.pdf), and Word (.docx).

### Embeddings (Google Gemini)
Embeddings convert text into high-dimensional vectors that capture semantic meaning.
The project uses the Google Gemini embedding model (models/embedding-001) to create
vectors for both documents and user queries.

### Language Model (Gemini)
The generative model (e.g. gemini-1.5-flash) produces the final natural-language
answer using the retrieved context. It is instructed to answer only from the provided
context and to say "I don't know" when the answer is not present.

### SQLite
SQLite is a self-contained, serverless relational database. It stores chat sessions
and message history, allowing conversations to persist between page reloads and be
reloaded from the session sidebar.

## Workflow

1. Upload one or more documents through the sidebar.
2. Each document is parsed and split into chunks.
3. Chunks are embedded and stored in the FAISS index.
4. A user asks a question, which is embedded and used to retrieve the most relevant chunks.
5. The chunks are passed to Gemini along with the question to generate a grounded answer.
6. Both the question and answer are saved to SQLite for the session.

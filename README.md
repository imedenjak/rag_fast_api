# RAG API 🤖

A Retrieval-Augmented Generation (RAG) API built with LangChain, ChromaDB, and FastAPI, containerized with Docker.

## Overview

This project loads content from a web page, splits it into chunks, stores embeddings in ChromaDB, and exposes a FastAPI endpoint to answer questions based on the retrieved context using OpenAI's GPT-3.5-turbo.

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — Vector store for embeddings
- **OpenAI** — Embeddings + LLM (GPT-3.5-turbo)
- **FastAPI** — REST API
- **Docker** — Containerization

## Project Structure

```
my_rag/
├── app/
│   ├── main.py          # FastAPI app
│   └── rag.py           # RAG chain logic
├── .env                 # API keys (never commit)
├── .env.example         # Template for API keys
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build instructions
└── README.md
```

## Prerequisites

- Docker Desktop
- OpenAI API key

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd my_rag
```

### 2. Set up environment variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-xxx
```

### 3. Build the Docker image

```bash
docker build -t my-rag-app .
```

### 4. Run the container

```bash
docker run -d --name rag-app -p 8000:8000 --env-file .env my-rag-app
```

### 5. Check container logs

```bash
docker logs -f rag-app
```

Wait until you see:
```
RAG chain ready!
```

## API Usage

### Swagger UI (Recommended)

Open your browser and navigate to:
```
http://localhost:8000/docs
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "chain_ready": true}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Task Decomposition?"}'
```

Response:
```json
{
  "question": "What is Task Decomposition?",
  "answer": "Task decomposition is the process of breaking down a complex task into smaller, more manageable steps..."
}
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is Task Decomposition?"}
)
print(response.json()["answer"])
```

## Docker Commands

```bash
# Stop container
docker stop rag-app

# Remove container
docker rm rag-app

# Rebuild after code changes
docker stop rag-app && docker rm rag-app
docker build -t my-rag-app .
docker run -d --name rag-app -p 8000:8000 --env-file .env my-rag-app

# View logs
docker logs -f rag-app
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |


## Notes

- The RAG chain is built **once at startup** — the web page is fetched and indexed when the container starts, not on every request.
- The container requires **outbound internet access** to reach the OpenAI API and load the source web page.
- First startup may take 30–60 seconds while the page is fetched and embeddings are generated.

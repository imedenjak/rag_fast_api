# RAG Agent

A Retrieval-Augmented Generation (RAG) agent built with LangChain, LangGraph, Qdrant, and Streamlit.

The agent generates multiple query variants, retrieves documents using reciprocal rank fusion, generates an answer, grades it for hallucinations, and retries with a rewritten question if the answer is not grounded.

## Tech Stack

- **LangGraph** — stateful agent with grading and retry logic
- **LangChain** — RAG pipeline orchestration
- **Qdrant** — persistent vector store
- **OpenAI** — configurable embeddings and chat models
- **Streamlit** — chat UI
- **LangSmith** — tracing and observability
- **Docker** — containerization

## Project Structure

```
adaptive_rag/
├── app/
│   ├── __init__.py         # makes app a Python package
│   ├── agent.py            # LangGraph agent — retrieve, generate, grade, retry
│   ├── chat_history.py     # SQLite-backed persistent chat history
│   ├── config.py           # env-based configuration
│   ├── ingest.py           # build Qdrant index (run once)
│   ├── logging_config.py   # structured logging (dev/json)
│   ├── rag.py              # multi-query + reciprocal rank fusion
│   └── streamlit_app.py    # Streamlit chat UI
├── eval/
│   ├── testset.json        # small evaluation dataset
│   └── evaluate.py         # RAGAS evaluation script
├── .env                    # API keys (never commit)
├── .env.example            # template
├── pyproject.toml          # dependencies and package metadata
├── langgraph.json          # LangGraph Studio config
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Agent Flow

```
User Question
      ↓
[retrieve]       multi-query generation + reciprocal rank fusion
      ↓
[generate]       answer from retrieved context
      ↓
[grade_answer]   is answer grounded in context?
      ↓
 grounded? ──YES──► END
     │
     NO
     └──► [rewrite_question] ──► [retrieve] ──► [generate]
                                      ↓ (max retries exceeded)
                                  [fallback]
```

## Prerequisites

- Python 3.12
- Docker Desktop
- OpenAI API key
- LangSmith API key (optional, for tracing)

## Getting Started

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd adaptive_rag
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start Qdrant and ingest documents

```bash
# Start Qdrant
docker compose up qdrant -d

# Build the vector index (run once, or after changing sources/models)
python -m app.ingest
```

### 4. Run locally

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Run with Docker Compose

```bash
# First run — build and start
docker compose up --build -d

# Ingest documents into the running container
docker compose exec app python -m app.ingest

# View logs
docker compose logs -f app
```

Open `http://localhost:8501`.

## LangGraph Studio (browser)

```bash
uv add langgraph-cli
langgraph dev
```

Open: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

## Environment Variables

| Variable                     | Required | Default                  | Description                                      |
|------------------------------|----------|--------------------------|--------------------------------------------------|
| `OPENAI_API_KEY`             | Yes      | —                        | OpenAI API key                                   |
| `OPENAI_CHAT_MODEL`          | No       | `gpt-4o-mini`            | Model for generation, grading, and rewriting     |
| `OPENAI_QUERY_MODEL`         | No       | same as chat model       | Model for generating query variants              |
| `OPENAI_EMBEDDING_MODEL`     | No       | `text-embedding-3-small` | Embedding model                                  |
| `OPENAI_EMBEDDING_DIMENSIONS`| No       | auto-detected            | Override vector size for custom embedding models |
| `QDRANT_URL`                 | No       | `http://localhost:6333`  | Qdrant server URL                                |
| `QDRANT_COLLECTION_NAME`     | No       | `rag_docs`               | Qdrant collection name                           |
| `LANGCHAIN_TRACING_V2`       | No       | `false`                  | Enable LangSmith tracing                         |
| `LANGCHAIN_API_KEY`          | No       | —                        | LangSmith API key                                |
| `LANGCHAIN_PROJECT`          | No       | —                        | LangSmith project name                           |

## RAGAS Evaluation

The `eval/` folder contains a small test set and an evaluation script that measures pipeline quality using [RAGAS](https://docs.ragas.io) — no human-labelled answers required.

**Metrics:**

| Metric | What it measures |
|---|---|
| **Answer Relevancy** | Is the answer about the question asked? |
| **Faithfulness** | Is every claim in the answer supported by retrieved chunks? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Does the retrieved context cover the reference answer? |

**Run evaluation:**

```bash
docker compose exec app python -m eval.evaluate
```

Results are printed to stdout and saved to `eval/results.json`.

**Interpreting scores** (all metrics are 0–1):
- `≥ 0.8` — good
- `0.5–0.8` — room for improvement
- `< 0.5` — investigate retrieval or generation quality

## Persistent Chat History

Chat history is stored in a SQLite database inside a named Docker volume (`chat_data`), so it survives container restarts. To inspect it:

```bash
docker compose exec app sqlite3 /data/chat_history.db "SELECT * FROM messages;"
```

## Notes

- Re-run `docker compose exec app python -m app.ingest <url>` after changing source URLs, chunk settings, or embedding model.
- Switching embedding models requires deleting the Qdrant collection first — vector dimensions must match.
- `@st.cache_resource` ensures the agent graph is built once per Streamlit session.
- Use `docker compose down -v` to also remove the chat history volume.

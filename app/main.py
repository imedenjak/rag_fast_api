from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import build_rag_chain
import uvicorn

load_dotenv()

app = FastAPI(title="RAG API")

# Build chain once at startup — not on every request
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("Loading RAG chain...")    # downloads page, builds vectorstore
    rag_chain = build_rag_chain()
    print("RAG chain ready!")

# Request/Response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# Routes
@app.get("/health")
def health():
    return {"status": "ok", "chain_ready": rag_chain is not None}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not ready yet")
    
    answer = rag_chain.invoke(request.question)
    
    return AnswerResponse(
        question=request.question,
        answer=answer
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
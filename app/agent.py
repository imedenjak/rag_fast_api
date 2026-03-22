from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from rag import build_retrieval_chain


# ------------------------- State ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    is_grounded: bool


# ------------------------- LLM ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Build once
retrieval_chain = build_retrieval_chain()


# ------------------------- Nodes ---------------------------------------------------------------------------
def retriever_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents using your existing RAG + rank fusion"""
    print("--- RETRIEVING ---")
    question = state["question"]
    # reciprocal_rank_fusion returns (doc, score) tuples — extract docs only
    results = retrieval_chain.invoke({"question": question})
    documents = [doc for doc, score in results]
    return {"documents": documents}


def generate_node(state: AgentState) -> AgentState:
    """Generate answer from retrieved documents"""
    print("--- GENERATING ---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template("""
Answer the following question based on this context:

{context}

Question: {question}
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer}


def grade_answer_node(state: AgentState) -> AgentState:
    """Check if answer is grounded in documents — no hallucination"""
    print("--- GRADING ANSWER ---")

    documents = state["documents"]
    answer = state["answer"]

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template("""
You are a grader checking if an answer is grounded in the provided context.
Answer ONLY 'yes' or 'no'.

Context: {context}
Answer: {answer}

Is the answer grounded in the context? (yes/no):
""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})
    is_grounded = result.strip().lower() == "yes"
    return {"is_grounded": is_grounded}


# ------------------------- Conditional Edge ------------------------------------------------------------
def should_retry(state: AgentState) -> str:
    """Decide whether to retry generation or finish"""
    if state["is_grounded"]:
        print("--- ANSWER GROUNDED, FINISHING ---")
        return "end"
    else:
        print("--- ANSWER NOT GROUNDED, RETRYING ---")
        return "generate"


# ------------------------- BUild Graph ------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_answer", grade_answer_node)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade_answer")

    # Conditional edge - retry if answer not grounded
    workflow.add_conditional_edges(
        "grade_answer",
        should_retry,
        {
            "end": END,
            "generate": "generate",  # ← loops back if not grounded
        },
    )

    return workflow.compile()

import os
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_docs"


def build_rag_chain():
    print("Loading Qdrant vectorstore...")

    embeddings = OpenAIEmbeddings()
    # embeddings = OllamaEmbeddings(model="llama3.2")
    client = QdrantClient(url=QDRANT_URL)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    print("Qdrant loaded successfully!")

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
                   If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                   Question: {question}
                   Context: {context}
                   Answer:""",
            )
        ]
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # llm = ChatOllama(model="llama3.2", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

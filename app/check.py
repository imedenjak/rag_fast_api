from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
info = client.get_collection("rag_docs")
print(info.config.params.vectors)
print(info.config.params.sparse_vectors)

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever


# Load embedding model
embedding_manager = EmbeddingManager()


# Connect to existing vector store
vector_store = VectorStoreManager()


# Create retriever
retriever = RAGRetriever(
    embedding_manager,
    vector_store
)


queries = [
    "What is Article 25 of the Constitution of Pakistan?",
    "What does Article 25 say about equality of citizens?",
    "What does Article 25 say about discrimination?"
]


for query in queries:

    print("\n" + "=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    results = retriever.retrieve(
        query,
        top_k=5
    )

    for result in results:

        print("\n" + "-" * 100)
        print("Rank:", result["rank"])
        print("Similarity:", result["similarity_score"])
        print("Distance:", result["distance"])
        print("Page:", result["metadata"].get("page"))
        print("Page label:", result["metadata"].get("page_label"))
        print("-" * 100)

        print(result["document"])


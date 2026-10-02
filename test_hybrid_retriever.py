from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever

from src.article_retriever import ArticleRetriever
from src.hybrid_retriever import HybridRetriever


# ----------------------------------------
# Semantic Retriever
# ----------------------------------------

embedding_manager = EmbeddingManager()

vector_store = VectorStoreManager()

semantic_retriever = RAGRetriever(
    embedding_manager=embedding_manager,
    vector_store=vector_store
)


# ----------------------------------------
# Article Retriever
# ----------------------------------------

article_retriever = ArticleRetriever()


# ----------------------------------------
# Hybrid Retriever
# ----------------------------------------

hybrid_retriever = HybridRetriever(
    semantic_retriever=semantic_retriever,
    article_retriever=article_retriever
)


# ----------------------------------------
# Test Queries
# ----------------------------------------

queries = [
    "What is Article 25?",
    "What does Article 25 say about equality?",
    "What does Article 25 say about discrimination?",
    "What protections are provided to women and children?"
]


for query in queries:

    print("\n" + "=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    results = hybrid_retriever.retrieve(
        query=query,
        top_k=3
    )

    if results:

        print(
            "Retrieval type:",
            results[0]["metadata"].get(
                "retrieval_type",
                "semantic"
            )
        )

        print(
            "Number of results:",
            len(results)
        )

        print("\nFirst result:\n")
        print(
            results[0]["document"][:1000]
        )

    else:

        print("No results found.")

        
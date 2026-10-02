from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever

from src.article_retriever import ArticleRetriever
from src.hybrid_retriever import HybridRetriever

from src.rag import RAGPipeline


# ----------------------------------------
# Semantic retrieval
# ----------------------------------------

embedding_manager = EmbeddingManager()

vector_store = VectorStoreManager()

semantic_retriever = RAGRetriever(
    embedding_manager=embedding_manager,
    vector_store=vector_store
)


# ----------------------------------------
# Article retrieval
# ----------------------------------------

article_retriever = ArticleRetriever()


# ----------------------------------------
# Hybrid retrieval
# ----------------------------------------

hybrid_retriever = HybridRetriever(
    semantic_retriever=semantic_retriever,
    article_retriever=article_retriever
)


# ----------------------------------------
# RAG pipeline
# ----------------------------------------

rag = RAGPipeline(
    retriever=hybrid_retriever
)


# ----------------------------------------
# Test
# ----------------------------------------

query = "What protections are provided to women and children?"

result = rag.generate_answer(
    query
)


print("\n" + "=" * 100)
print("QUESTION")
print("=" * 100)

print(query)


print("\n" + "=" * 100)
print("ANSWER")
print("=" * 100)

print(result["answer"])


print("\n" + "=" * 100)
print("RETRIEVAL TYPE")
print("=" * 100)

if result["sources"]:

    print(
        result["sources"][0]["retrieval_type"]
    )
    
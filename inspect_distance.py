import numpy as np

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager


# Load embedding model
embedding_manager = EmbeddingManager()


# Connect to existing Chroma database
vector_store = VectorStoreManager()


query = "All citizens are equal before law"


# Generate query embedding
query_embedding = embedding_manager.model.encode(
    [query]
)[0]


# Query Chroma directly
results = vector_store.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
    include=["documents", "metadatas", "distances", "embeddings"]
)


print("\n" + "=" * 100)
print("DISTANCE METRIC DIAGNOSTIC")
print("=" * 100)

for i in range(len(results["documents"][0])):

    document = results["documents"][0][i]
    distance = results["distances"][0][i]
    stored_embedding = np.array(results["embeddings"][0][i])

    cosine_similarity = np.dot(
        query_embedding,
        stored_embedding
    ) / (
        np.linalg.norm(query_embedding)
        * np.linalg.norm(stored_embedding)
    )

    print("\n" + "-" * 100)
    print("Rank:", i + 1)
    print("Page:", results["metadatas"][0][i].get("page"))
    print("Chroma distance:", distance)
    print("1 - distance:", 1 - distance)
    print("Manual cosine similarity:", cosine_similarity)
    print("-" * 100)

    print(document[:500])


    
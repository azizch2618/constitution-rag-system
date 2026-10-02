import chromadb

from src.embeddings import EmbeddingManager


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

embedding_manager = EmbeddingManager()


# --------------------------------------------------
# 2. Connect to Chroma
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="data/vector_store"
)

collection = client.get_collection(
    name="pdf_documents_cosine"
)

print("\nCollection:", collection.name)
print("Documents:", collection.count())


# --------------------------------------------------
# 3. Test query
# --------------------------------------------------

query = "All citizens are equal before law"

query_embedding = embedding_manager.model.encode(
    [query]
)[0]


results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("\n" + "=" * 100)
print("COSINE RETRIEVAL TEST")
print("=" * 100)

for i in range(len(results["documents"][0])):

    distance = results["distances"][0][i]

    # For cosine distance:
    similarity = 1 - distance

    print("\n" + "-" * 100)
    print("Rank:", i + 1)
    print("Page:", results["metadatas"][0][i].get("page"))
    print("Page label:", results["metadatas"][0][i].get("page_label"))
    print("Cosine distance:", distance)
    print("Cosine similarity:", similarity)
    print("-" * 100)

    print(results["documents"][0][i])


    
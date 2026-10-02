from src.loader import load_all_pdfs
from src.chunking import split_docs
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager


print("Loading PDF documents...")

documents = load_all_pdfs()

print(f"Pages loaded: {len(documents)}")


print("\nSplitting documents into chunks...")

chunks = split_docs(documents)

print(f"Chunks created: {len(chunks)}")


print("\nLoading embedding model...")

embedding_manager = EmbeddingManager()


print("\nGenerating embeddings...")

texts = [doc.page_content for doc in chunks]

embeddings = embedding_manager.generate_embeddings(texts)


print("\nCreating vector store...")

vector_store = VectorStoreManager()


print("\nAdding documents to vector store...")

vector_store.add_documents(
    chunks,
    embeddings
)


print("\n================================")
print("INGESTION COMPLETE")
print("================================")

print("Pages:", len(documents))
print("Chunks:", len(chunks))
print(
    "Vectors in Chroma:",
    vector_store.collection.count()
)


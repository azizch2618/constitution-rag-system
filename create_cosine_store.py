import os
import uuid
import chromadb

from src.loader import load_all_pdfs
from src.embeddings import EmbeddingManager
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

documents = load_all_pdfs()

print("Pages loaded:", len(documents))


# --------------------------------------------------
# 2. Split documents
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

embedding_manager = EmbeddingManager()


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

texts = [chunk.page_content for chunk in chunks]

print("\nGenerating embeddings...")

embeddings = embedding_manager.model.encode(
    texts,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 5. Create NEW Chroma collection
# --------------------------------------------------

persist_directory = "data/vector_store"

client = chromadb.PersistentClient(
    path=persist_directory
)

collection = client.get_or_create_collection(
    name="pdf_documents_cosine",
    metadata={
        "hnsw:space": "cosine",
        "description": "Constitution RAG collection using cosine distance"
    }
)

print("\nCollection created:")
print(collection.name)

print("Existing documents:", collection.count())


# --------------------------------------------------
# 6. Add documents
# --------------------------------------------------

if collection.count() == 0:

    ids = []
    metadatas = []
    documents_content = []
    embeddings_list = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        ids.append(f"cosine_doc_{uuid.uuid4()}")

        metadata = dict(chunk.metadata)

        metadata["doc_index"] = i
        metadata["content_length"] = len(
            chunk.page_content
        )

        metadatas.append(metadata)

        documents_content.append(
            chunk.page_content
        )

        embeddings_list.append(
            embedding.tolist()
        )

    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents_content,
        embeddings=embeddings_list
    )

    print("\nDocuments added:", len(ids))

else:

    print("\nCollection already contains documents.")
    print("Skipping insertion.")


print(
    "\nFinal collection count:",
    collection.count()
)


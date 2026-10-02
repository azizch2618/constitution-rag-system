import os
import uuid
import chromadb


class VectorStoreManager:

    def __init__(
        self,
        persist_directory="data/vector_store",
        collection_name="pdf_documents_cosine"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):

        os.makedirs(self.persist_directory, exist_ok=True)

        # Create / connect to Chroma database
        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Get existing cosine collection or create it
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "description": "Vector store collection for PDF embeddings using cosine distance"
            }
        )

        print(
            "Initialized vector store with collection:",
            self.collection_name
        )

        print(
            "Documents in collection:",
            self.collection.count()
        )

    def add_documents(self, documents, embeddings):

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents does not match number of embeddings"
            )

        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            doc_id = f"doc_{uuid.uuid4()}"

            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            all_metadata.append(metadata)
            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )

        print(
            "Total documents added:",
            len(documents_content)
        )

        print(
            "Documents in collection:",
            self.collection.count()
        )
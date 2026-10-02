from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager


class RAGRetriever:

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStoreManager
    ):

        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(
        self,
        query,
        top_k=5,
        score_threshold=0.0
    ):

        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )[0]

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:

            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i, (
                doc_id,
                metadata,
                document,
                distance
            ) in enumerate(
                zip(
                    ids,
                    metadatas,
                    documents,
                    distances
                )
            ):

                cosine_similarity = 1 - distance

                if cosine_similarity >= score_threshold:

                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "metadata": metadata,
                            "document": document,
                            "distance": distance,
                            "similarity_score": cosine_similarity,
                            "rank": i + 1,
                            "retrieval_type": "semantic"
                        }
                    )

        return retrieved_docs
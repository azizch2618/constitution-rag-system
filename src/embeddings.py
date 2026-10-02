from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name

        print("Loading Model....", self.model_name)

        self.model = SentenceTransformer(self.model_name)

        print(
            "Embedding Dimensions =",
            self.model.get_embedding_dimension()
        )

    def generate_embeddings(self, text):
        embeddings = self.model.encode(
            text,
            show_progress_bar=True
        )

        print("Embeddings Shape:", embeddings.shape)

        return embeddings
    
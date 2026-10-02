from src.hybrid_retriever import HybridRetriever
from src.llm import get_llm


class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRetriever
    ):

        self.retriever = retriever
        self.llm = get_llm()

    def generate_answer(
        self,
        query,
        top_k=3
    ):

        results = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )

        if not results:

            return {
                "answer": "I could not find relevant information in the Constitution.",
                "sources": []
            }

        context = "\n\n".join(
            result["document"]
            for result in results
        )

        prompt = f"""
You are a question-answering assistant for the Constitution of Pakistan.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer the question,
say that the information was not found in the retrieved context.

Do not invent constitutional provisions.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": results
        }
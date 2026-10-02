import re

from src.article_retriever import ArticleRetriever
from src.retriever import RAGRetriever
from src.context_filter import remove_following_articles


class HybridRetriever:

    def __init__(
        self,
        semantic_retriever: RAGRetriever,
        article_retriever: ArticleRetriever
    ):
        self.semantic_retriever = semantic_retriever
        self.article_retriever = article_retriever

    def detect_article_number(self, query):

        pattern = r"\barticle\s+(\d+[A-Za-z]?)\b"

        match = re.search(
            pattern,
            query,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

        return None

    def retrieve(
        self,
        query,
        top_k=5,
        score_threshold=0.0
    ):

        # ----------------------------------------
        # 1. Detect Article reference
        # ----------------------------------------

        article_number = self.detect_article_number(
            query
        )

        # ----------------------------------------
        # 2. Article-aware retrieval
        # ----------------------------------------

        if article_number:

            results = self.article_retriever.retrieve(
                article_number
            )

            if results:
                return results

        # ----------------------------------------
        # 3. Semantic retrieval
        # ----------------------------------------

        results = self.semantic_retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # ----------------------------------------
        # 4. Context cleaning
        # ----------------------------------------

        for result in results:

            result["document"] = remove_following_articles(
                result["document"]
            )

        return results
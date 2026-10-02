import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader import load_all_pdfs


class ArticleRetriever:

    def __init__(self):

        documents = load_all_pdfs()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        self.chunks = splitter.split_documents(documents)

    def find_article_start(self, article_number):

        pattern = (
            rf"^{re.escape(article_number)}\.\s*$"
            rf".*?"
            rf"^\(1\)"
        )

        matches = []

        for i, chunk in enumerate(self.chunks):

            if re.search(
                pattern,
                chunk.page_content,
                flags=re.MULTILINE | re.DOTALL
            ):

                matches.append(i)

        return matches

    def extract_article(self, article_number):

        start_matches = self.find_article_start(
            article_number
        )

        if not start_matches:
            return None

        start_chunk_index = start_matches[0]

        article_text = []

        for i in range(
            start_chunk_index,
            len(self.chunks)
        ):

            text = self.chunks[i].page_content

            # Remove everything before the actual
            # article heading in the first chunk.
            if i == start_chunk_index:

                pattern = (
                    rf"^{re.escape(article_number)}\.\s*$"
                )

                match = re.search(
                    pattern,
                    text,
                    flags=re.MULTILINE
                )

                if match:
                    text = text[match.start():]

            # Find the next article heading.
            next_article_pattern = (
                r"(?m)^\d+[A-Za-z]*\."
            )

            next_match = re.search(
                next_article_pattern,
                text
            )

            if i > start_chunk_index and next_match:

                text = text[:next_match.start()]

                # Remove PDF footnote artifacts
                # at the article boundary.
                text = re.sub(
                    r"\s*\d+\s*\[\s*$",
                    "",
                    text
                )

                article_text.append(text)

                break

            article_text.append(text)

        return "\n".join(article_text).strip()

    def retrieve(self, article_number):

        article = self.extract_article(
            article_number
        )

        if not article:
            return []

        return [
            {
                "document": article,
                "metadata": {
                    "article": article_number
                },
                "retrieval_type": "article",
                "similarity_score": 1.0,
                "rank": 1
            }
        ]
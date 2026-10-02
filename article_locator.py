import re

from src.loader import load_all_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_chunks():

    documents = load_all_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


def find_article_start(chunks, article_number):

    pattern = (
        rf"^{re.escape(article_number)}\.\s*$"
        rf".*?"
        rf"^\(1\)"
    )

    matches = []

    for i, chunk in enumerate(chunks):

        if re.search(
            pattern,
            chunk.page_content,
            flags=re.MULTILINE | re.DOTALL
        ):

            matches.append(i)

    return matches


def extract_article(chunks, article_number):

    start_matches = find_article_start(
        chunks,
        article_number
    )

    if not start_matches:
        return None

    start_chunk_index = start_matches[0]

    article_text = []

    for i in range(start_chunk_index, len(chunks)):

        text = chunks[i].page_content

        # For the first chunk, remove everything
        # before the actual article heading.
        if i == start_chunk_index:

            pattern = rf"^{re.escape(article_number)}\.\s*$"

            match = re.search(
                pattern,
                text,
                flags=re.MULTILINE
            )

            if match:
                text = text[match.start():]

        # Look for the next article heading.
        next_article_pattern = r"(?m)^\d+[A-Za-z]*\."

        next_match = re.search(
            next_article_pattern,
            text
        )

        # We only want a next article if it is NOT
        # the article we are currently extracting.
        if i > start_chunk_index and next_match:

            text = text[:next_match.start()]

            # Remove PDF footnote artifacts at the boundary.
            text = re.sub(
                r"\s*\d+\s*\[\s*$",
                "",
                text
            )

            article_text.append(text)
            break

        article_text.append(text)

    return "\n".join(article_text).strip()


if __name__ == "__main__":

    chunks = load_chunks()

    article_number = "25"

    article = extract_article(
        chunks,
        article_number
    )

    print("\n" + "=" * 100)
    print(f"EXTRACTED ARTICLE {article_number}")
    print("=" * 100)

    if article:
        print(article)
    else:
        print("Article not found.")
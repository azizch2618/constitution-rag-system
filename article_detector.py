import re


def detect_article_number(query):

    pattern = r"\barticle\s+(\d+[A-Za-z]?)\b"

    match = re.search(
        pattern,
        query,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None

queries = [
    "What is Article 25?",
    "Explain Article 25",
    "What does article 25 say about equality?",
    "Tell me about Article 25A",
    "What does the constitution say about equality?"
]

for query in queries:

    article = detect_article_number(query)

    print(
        f"Query: {query}"
    )

    print(
        f"Detected Article: {article}"
    )

    print("-" * 60)


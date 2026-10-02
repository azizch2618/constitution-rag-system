import re


def remove_following_articles(text):

    # Find an article heading such as:
    # 25A.
    # 26.
    #
    # The heading must be at the beginning of a line.

    pattern = r"(?m)^\d+[A-Za-z]+\."

    match = re.search(
        pattern,
        text
    )

    if not match:
        return text.strip()

    cleaned = text[:match.start()]

    # Remove PDF footnote artifacts immediately
    # before the next article heading.
    cleaned = re.sub(
        r"\s*\d+\s*\[\s*$",
        "",
        cleaned
    )

    return cleaned.strip()
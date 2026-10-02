from src.article_retriever import ArticleRetriever


retriever = ArticleRetriever()

result = retriever.retrieve("25")


if result:

    print("\n" + "=" * 100)
    print("ARTICLE RETRIEVAL RESULT")
    print("=" * 100)

    print(
        result[0]["document"]
    )

    print("\nMetadata:")
    print(
        result[0]["metadata"]
    )

else:

    print("Article not found.")



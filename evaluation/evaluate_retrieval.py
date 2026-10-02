import json

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.article_retriever import ArticleRetriever
from src.hybrid_retriever import HybridRetriever


# ============================================================
# LOAD TEST DATA
# ============================================================

with open(
    "evaluation/test_questions.json",
    "r",
    encoding="utf-8"
) as file:

    test_questions = json.load(file)


# ============================================================
# INITIALIZE RETRIEVERS
# ============================================================

print("\nLoading RAG retrieval system...\n")

embedding_manager = EmbeddingManager()

vector_store = VectorStoreManager()

semantic_retriever = RAGRetriever(
    embedding_manager=embedding_manager,
    vector_store=vector_store
)

article_retriever = ArticleRetriever()

hybrid_retriever = HybridRetriever(
    semantic_retriever=semantic_retriever,
    article_retriever=article_retriever
)


# ============================================================
# EVALUATION
# ============================================================

results = []

retrieval_type_correct = 0
retrieval_content_correct = 0


for index, test_case in enumerate(
    test_questions,
    start=1
):

    question = test_case["question"]

    expected_type = test_case[
        "expected_retrieval_type"
    ]

    print("\n" + "=" * 80)

    print(
        f"TEST {index}/{len(test_questions)}"
    )

    print("=" * 80)

    print(
        "Question:",
        question
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    retrieved = hybrid_retriever.retrieve(
        query=question,
        top_k=3
    )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not retrieved:

        print("Result: NO RETRIEVAL")

        results.append(
            {
                "question": question,
                "expected_type": expected_type,
                "actual_type": None,
                "retrieval_type_correct": False,
                "content_correct": False
            }
        )

        continue

    # --------------------------------------------------------
    # Top result
    # --------------------------------------------------------

    top_result = retrieved[0]

    actual_type = top_result.get(
        "retrieval_type"
    )

    document = top_result.get(
        "document",
        ""
    )

    document_lower = document.lower()

    # --------------------------------------------------------
    # Check retrieval type
    # --------------------------------------------------------

    type_correct = (
        actual_type == expected_type
    )

    if type_correct:

        retrieval_type_correct += 1

    # --------------------------------------------------------
    # Check content
    # --------------------------------------------------------

    content_correct = False

    # ========================================================
    # ARTICLE RETRIEVAL
    # ========================================================

    if expected_type == "article":

        expected_article = test_case[
            "expected_article"
        ]

        actual_article = top_result.get(
            "metadata",
            {}
        ).get(
            "article"
        )

        content_correct = (
            actual_article == expected_article
        )

        print(
            "Expected article:",
            expected_article
        )

        print(
            "Actual article:",
            actual_article
        )

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    elif expected_type == "semantic":

        expected_keywords = test_case[
            "expected_keywords"
        ]

        matched_keywords = []

        for keyword in expected_keywords:

            if keyword.lower() in document_lower:

                matched_keywords.append(
                    keyword
                )

        content_correct = (
            len(matched_keywords) > 0
        )

        print(
            "Expected keywords:",
            expected_keywords
        )

        print(
            "Matched keywords:",
            matched_keywords
        )

    # --------------------------------------------------------
    # Count content correctness
    # --------------------------------------------------------

    if content_correct:

        retrieval_content_correct += 1

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        "Expected type:",
        expected_type
    )

    print(
        "Actual type:",
        actual_type
    )

    print(
        "Type check:",
        "PASS"
        if type_correct
        else "FAIL"
    )

    print(
        "Content check:",
        "PASS"
        if content_correct
        else "FAIL"
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "question": question,
            "expected_type": expected_type,
            "actual_type": actual_type,
            "retrieval_type_correct": type_correct,
            "content_correct": content_correct
        }
    )


# ============================================================
# SUMMARY
# ============================================================

total_tests = len(test_questions)

retrieval_type_accuracy = (
    retrieval_type_correct / total_tests
) * 100

content_accuracy = (
    retrieval_content_correct / total_tests
) * 100


print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print(
    f"Total tests: {total_tests}"
)

print(
    f"Retrieval type accuracy: "
    f"{retrieval_type_accuracy:.2f}%"
)

print(
    f"Retrieval content accuracy: "
    f"{content_accuracy:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "evaluation/results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nDetailed results saved to:"
)

print(
    "evaluation/results.json"
)
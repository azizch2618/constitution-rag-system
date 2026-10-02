import streamlit as st

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.article_retriever import ArticleRetriever
from src.hybrid_retriever import HybridRetriever
from src.rag import RAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pakistan Constitution Agent",
    page_icon="🇵🇰",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("🇵🇰 Pakistan Constitution")

st.caption(
    "AI-powered question answering over the Constitution of Pakistan"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application uses Retrieval-Augmented Generation (RAG)
        to answer questions from the Constitution of Pakistan.
        """
    )

    st.divider()

    st.subheader("Architecture")

    st.write(
        """
        PDF → Chunking → Embeddings → ChromaDB

        ↓

        Hybrid Retrieval

        ↓

        Groq LLM

        ↓

        Answer
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# LOAD RAG PIPELINE
# ============================================================

@st.cache_resource
def load_rag_pipeline():

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_manager = EmbeddingManager()

    # --------------------------------------------------------
    # Existing ChromaDB vector store
    # --------------------------------------------------------

    vector_store = VectorStoreManager()

    # --------------------------------------------------------
    # Semantic retriever
    # --------------------------------------------------------

    semantic_retriever = RAGRetriever(
        embedding_manager=embedding_manager,
        vector_store=vector_store
    )

    # --------------------------------------------------------
    # Article retriever
    # --------------------------------------------------------

    article_retriever = ArticleRetriever()

    # --------------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        article_retriever=article_retriever
    )

    # --------------------------------------------------------
    # Complete RAG pipeline
    # --------------------------------------------------------

    rag = RAGPipeline(
        retriever=hybrid_retriever
    )

    return rag, vector_store


# ============================================================
# INITIALIZE
# ============================================================

with st.spinner("Loading RAG system..."):

    rag, vector_store = load_rag_pipeline()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.success(
    f"RAG system ready • {vector_store.collection.count()} "
    "documents in vector store"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Display retrieval information for assistant
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            sources = message["sources"]

            source = sources[0]

            with st.expander(
                "🔎 Retrieval information"
            ):

                retrieval_type = source.get(
                    "retrieval_type"
                )

                st.write(
                    "Retrieval type:",
                    retrieval_type
                )

                # --------------------------------------------
                # Article retrieval
                # --------------------------------------------

                if retrieval_type == "article":

                    article_number = source.get(
                        "metadata",
                        {}
                    ).get(
                        "article"
                    )

                    if article_number:

                        st.write(
                            "Article:",
                            article_number
                        )

                # --------------------------------------------
                # Semantic retrieval
                # --------------------------------------------

                else:

                    similarity = source.get(
                        "similarity_score"
                    )

                    if similarity is not None:

                        st.write(
                            "Top similarity:",
                            round(
                                similarity,
                                4
                            )
                        )

                # --------------------------------------------
                # Retrieved context
                # --------------------------------------------

                st.divider()

                st.write(
                    "Retrieved context:"
                )

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"**Source {i}**"
                    )

                    st.write(
                        source["document"]
                    )

                    if (
                        source.get(
                            "similarity_score"
                        ) is not None
                    ):

                        st.caption(
                            "Similarity: "
                            + str(
                                round(
                                    source[
                                        "similarity_score"
                                    ],
                                    4
                                )
                            )
                        )

                    st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about the Constitution..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(query)

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the Constitution..."
        ):

            try:

                result = rag.generate_answer(
                    query=query,
                    top_k=3
                )

                answer = result["answer"]

                sources = result["sources"]

                # --------------------------------------------
                # Display answer
                # --------------------------------------------

                st.markdown(answer)

                # --------------------------------------------
                # Retrieval information
                # --------------------------------------------

                if sources:

                    source = sources[0]

                    with st.expander(
                        "🔎 Retrieval information"
                    ):

                        retrieval_type = source.get(
                            "retrieval_type"
                        )

                        st.write(
                            "Retrieval type:",
                            retrieval_type
                        )

                        if (
                            retrieval_type == "article"
                        ):

                            article_number = source.get(
                                "metadata",
                                {}
                            ).get(
                                "article"
                            )

                            if article_number:

                                st.write(
                                    "Article:",
                                    article_number
                                )

                        else:

                            similarity = source.get(
                                "similarity_score"
                            )

                            if similarity is not None:

                                st.write(
                                    "Top similarity:",
                                    round(
                                        similarity,
                                        4
                                    )
                                )

                        # ------------------------------------
                        # Context
                        # ------------------------------------

                        st.divider()

                        st.write(
                            "Retrieved context:"
                        )

                        for i, source in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"**Source {i}**"
                            )

                            st.write(
                                source["document"]
                            )

                            if (
                                source.get(
                                    "similarity_score"
                                ) is not None
                            ):

                                st.caption(
                                    "Similarity: "
                                    + str(
                                        round(
                                            source[
                                                "similarity_score"
                                            ],
                                            4
                                        )
                                    )
                                )

                            st.divider()

                # --------------------------------------------
                # Save assistant message
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                error_message = (
                    "Sorry, an error occurred while "
                    "processing your question."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )
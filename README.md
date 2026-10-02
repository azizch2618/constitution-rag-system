# 🇵🇰 Constitution of Pakistan — RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows users to ask questions about the Constitution of Pakistan (1973) using natural language.

The system combines semantic vector retrieval, article-aware retrieval, context filtering, and an LLM to generate answers grounded in the retrieved constitutional text.

---

## 📌 Project Overview

This project demonstrates how a RAG pipeline can be built over a large PDF document.

Instead of sending the complete Constitution to the LLM, the system:

1. Loads the Constitution PDF
2. Extracts the document text
3. Splits the text into smaller chunks
4. Generates vector embeddings
5. Stores embeddings in ChromaDB
6. Retrieves relevant information for a user query
7. Detects explicit Article references
8. Uses hybrid retrieval when appropriate
9. Filters unnecessary following-article content
10. Sends the retrieved context to an LLM
11. Generates a grounded answer
12. Displays the result through a Streamlit interface

---

## 🧠 RAG Architecture

```text
                 Constitution of Pakistan PDF
                            │
                            ▼
                    PDF Text Extraction
                            │
                            ▼
                    Recursive Chunking
                            │
                            ▼
                  Sentence Transformer
                     Embeddings
                            │
                            ▼
                       ChromaDB
                  Cosine Vector Store
                            │
                     User Question
                            │
                     ┌──────┴──────┐
                     │             │
              Article Query    Normal Query
                     │             │
                     ▼             ▼
             Article Retriever  Semantic
                                Retriever
                     │             │
                     └──────┬──────┘
                            ▼
                    Hybrid Retriever
                            │
                            ▼
                    Context Filtering
                            │
                            ▼
                       Groq LLM
                            │
                            ▼
                      Final Answer
                            │
                            ▼
                     Streamlit UI
🚀 Features
Semantic Retrieval

Uses sentence-transformer embeddings to find constitution passages that are semantically related to the user's question.

Article-Aware Retrieval

When a user explicitly asks about an article, the system detects the article number and retrieves the corresponding constitutional article directly.

Examples:

What is Article 25?
What does Article 25 say about equality?
What is Article 25A?
What does Article 26 say about discrimination?
Hybrid Retrieval

The system combines:

Article-aware retrieval for explicit Article queries
Semantic retrieval for general natural-language questions
Context Filtering

Retrieved chunks can contain the beginning of a following article because of document chunk boundaries.

A context filtering layer removes unnecessary following-article content before sending the context to the LLM.

LLM Answer Generation

Retrieved constitutional context is provided to a Groq-hosted LLM.

The prompt instructs the model to answer using only the retrieved context and avoid inventing constitutional provisions.

Streamlit Interface

The project includes a Streamlit chat interface for interacting with the RAG system.

🛠️ Tech Stack
Python
LangChain
Sentence Transformers
ChromaDB
Groq
LangChain Groq
Streamlit
PyPDFLoader
RecursiveCharacterTextSplitter
📂 Project Structure
RAG PROJECT/
│
├── data/
│   ├── pdfs/
│   └── vector_store/
│
├── evaluation/
│   ├── test_questions.json
│   └── evaluate_retrieval.py
│
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── article_retriever.py
│   ├── context_filter.py
│   ├── hybrid_retriever.py
│   ├── llm.py
│   └── rag.py
│
├── app.py
├── ingest.py
├── test_rag.py
├── test_retrieval.py
├── test_article_retriever.py
├── test_context_filter.py
├── test_hybrid_retriever.py
├── test_cosine_retrieval.py
├── article_detector.py
├── article_locator.py
├── find_article.py
├── inspect_chunks.py
├── inspect_article_chunks.py
├── inspect_distance.py
├── create_cosine_store.py
├── requirements.txt
├── .gitignore
└── README.md
⚙️ Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-PROJECT
2. Create and activate the virtual environment

The project can be run inside a Python virtual environment.

Example:

python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure the Groq API key

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here

Never commit the .env file.

📄 Add the Constitution PDF

Place the Constitution PDF inside:

data/pdfs/

The source PDF is intentionally not included in the repository.

🔄 Build the Vector Store

Run the ingestion pipeline:

python ingest.py

This processes the PDF, creates embeddings, and stores them in ChromaDB.

🧪 Test the RAG Pipeline

Run:

python test_rag.py

This tests the complete retrieval and answer-generation pipeline.

📊 Run Retrieval Evaluation

Run:

python -m evaluation.evaluate_retrieval

Current evaluation dataset contains six test questions covering:

Article retrieval
Semantic retrieval
Article 25
Article 25A
Article 26
Freedom of speech
Women and children protections

Current baseline evaluation:

Retrieval Type Accuracy: 100%
Retrieval Content Accuracy: 100%

This is a small smoke-test evaluation set and should not be interpreted as a comprehensive benchmark.

💻 Run the Streamlit Application

Start the application:

streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

Example:

http://localhost:8501
💬 Example Questions
What is Article 25?

What does Article 25 say about equality?

What is Article 25A?

What does Article 26 say about discrimination?

What does the Constitution say about freedom of speech?

What protections are provided to women and children?
🔍 Retrieval Strategy

The system uses two retrieval strategies.

1. Article Retrieval

If the question contains an explicit article reference:

Article 25
Article 25A
Article 26

the system attempts direct article-aware retrieval.

2. Semantic Retrieval

For general questions, the query is converted into an embedding and compared against the stored document embeddings.

The project uses cosine distance in ChromaDB.

📈 Evaluation

The initial evaluation produced:

Total Tests: 6

Retrieval Type Accuracy: 100.00%

Retrieval Content Accuracy: 100.00%

The evaluation currently focuses on retrieval behavior rather than being a complete end-to-end benchmark of generated answer quality.

🔐 Security

API credentials are stored in environment variables.

The following files are excluded from Git:

.env
PAK_CONSTITUTION.ipynb
.ipynb_checkpoints/

The original source PDF and generated ChromaDB vector store are also excluded from the repository.

Never expose API keys in source code, notebooks, screenshots, or Git history.

🎯 Learning Objectives

This project demonstrates practical understanding of:

RAG architecture
Document ingestion
PDF text extraction
Text chunking
Embeddings
Vector databases
Cosine similarity
Semantic search
Metadata-based retrieval
Hybrid retrieval
Context filtering
LLM prompting
Streamlit application development
Retrieval evaluation
Environment-variable based API security
🔮 Future Improvements

Potential future improvements include:

Larger retrieval evaluation datasets
Recall@K
MRR
Context precision
End-to-end answer evaluation
Better conversational memory
Improved citation/source display
Reranking
Metadata filtering
More robust article extraction
Deployment
👨‍💻 Author

Muhammad Aziz Ullah

AI/ML Developer | Python Developer | Web & Application Developer
⚠️ Disclaimer

This project is an educational RAG implementation. Answers are generated from retrieved document context and should not be treated as legal advice. 

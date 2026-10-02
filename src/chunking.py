from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(documents, chunk_size=500, chunk_overlap=50):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = text_splitter.split_documents(documents)

    return chunked_docs

from langchain_core.documents import Document
import os
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader


def load_all_pdfs():
    folder_path = "data/pdfs"
    num_docs = 0 
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            
            pdf_path = os.path.join(folder_path, filename)

            loader = PyPDFLoader(pdf_path)
            doc = loader.load()
            
            all_docs.extend(doc)
            num_docs += 1

    print('total pdfs:', num_docs)
    print('total pages:', len(all_docs))

    return all_docs
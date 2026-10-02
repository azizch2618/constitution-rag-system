from src.loader import load_all_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter


documents = load_all_pdfs()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# Inspect chunks around Article 25
start = 103
end = 110

for i in range(start, end + 1):

    chunk = chunks[i]

    print("\n" + "=" * 100)
    print(f"CHUNK INDEX : {i}")
    print(f"PAGE        : {chunk.metadata.get('page')}")
    print(f"PAGE LABEL  : {chunk.metadata.get('page_label')}")
    print(f"LENGTH      : {len(chunk.page_content)}")
    print("=" * 100)

    print(chunk.page_content)
    
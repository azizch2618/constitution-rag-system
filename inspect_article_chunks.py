from src.loader import load_all_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter


documents = load_all_pdfs()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


article_number = "25"

print("\n" + "=" * 100)
print(f"SEARCHING FOR ARTICLE {article_number}")
print("=" * 100)


for i, chunk in enumerate(chunks):

    text = chunk.page_content

    if (
        f"\n{article_number}.\n" in text
        or f"\n{article_number}.\r\n" in text
        or f"{article_number}.\n(1)" in text
    ):

        print("\n" + "-" * 100)
        print("CHUNK:", i)
        print("PAGE:", chunk.metadata.get("page"))
        print("PAGE LABEL:", chunk.metadata.get("page_label"))
        print("-" * 100)

        print(chunk.page_content)
        
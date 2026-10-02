from src.loader import load_all_pdfs


documents = load_all_pdfs()

search_terms = [
    "All citizens are equal before law",
    "equal protection of law",
    "There shall be no discrimination"
]


for term in search_terms:

    print("\n" + "=" * 80)
    print(f"SEARCHING: {term}")
    print("=" * 80)

    found = False

    for i, doc in enumerate(documents):

        if term.lower() in doc.page_content.lower():

            print(f"\nFOUND!")
            print(f"Document index : {i}")
            print(f"Page           : {doc.metadata.get('page')}")
            print(f"Page label     : {doc.metadata.get('page_label')}")

            print("\nCONTENT:")
            print(doc.page_content)

            found = True
            break

    if not found:
        print("Not found.")
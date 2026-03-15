from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(pages)

    return docs
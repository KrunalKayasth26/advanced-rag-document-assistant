import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"

def create_vector_store(docs):

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # If index already exists → load it
    if os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH):

        print("⚡ Loading existing FAISS index...")

        vector_store = FAISS.load_local(
            INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    else:

        print("🔄 Creating FAISS index for first time...")

        vector_store = FAISS.from_documents(docs, embedding_model)

        os.makedirs(INDEX_PATH, exist_ok=True)

        vector_store.save_local(INDEX_PATH)

        print("✅ FAISS index saved")

    return vector_store
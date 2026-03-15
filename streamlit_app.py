import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from utils.context_compression import compress_context
from utils.adaptive_retrieval import choose_k
from utils.analytics import RAGAnalytics
from utils.query_expansion import expand_query
from utils.pdf_loader import load_pdf
from utils.chunking import chunk_text
from utils.embeddings import create_vector_store
from utils.semantic_cache import SemanticCache

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from langchain_community.retrievers import BM25Retriever


st.set_page_config(page_title="Document Intelligence System", layout="wide")

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "cache" not in st.session_state:
    st.session_state.cache = SemanticCache()

if "analytics" not in st.session_state:
    st.session_state.analytics = RAGAnalytics()


st.title("📄 AI Document Intelligence Assistant")


uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:

    with st.spinner("Processing documents..."):

        all_pages = []

        for file in uploaded_files:

            path = f"temp_{file.name}"

            with open(path, "wb") as f:
                f.write(file.read())

            pages = load_pdf(path)

            all_pages.extend(pages)

        docs = chunk_text(all_pages)

        vector_store = create_vector_store(docs)

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = 10

        st.session_state.vector_store = vector_store
        st.session_state.bm25 = bm25

        st.success("PDFs processed successfully")


query = st.chat_input("Ask something about your documents...")


if query and st.session_state.vector_store:

    cached_answer = st.session_state.cache.search(query)

    if cached_answer:

        st.chat_message("user").write(query)
        st.chat_message("assistant").write(cached_answer)

        st.info("⚡ Answer returned from semantic cache")

        st.stop()


    expanded_queries = expand_query(client, query)

    results = []

    k = choose_k(query)
    st.caption(f"Adaptive retrieval using k = {k}")

    for q in expanded_queries:

        faiss_results = st.session_state.vector_store.similarity_search(q, k=k)
        bm25_results = st.session_state.bm25.invoke(q)

        results.extend(faiss_results)
        results.extend(bm25_results)


    unique = {}

    for doc in results:
        unique[doc.page_content] = doc

    results = list(unique.values())


    pairs = [(query, doc.page_content) for doc in results]

    scores = reranker.predict(pairs)

    scored = list(zip(results, scores))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = [doc for doc, score in scored[:3]]


    if not results:
        st.warning("No relevant information found.")
        st.stop()


    sources = set()

    for doc in results:
        page = doc.metadata.get("page", 0) + 1
        sources.add(page)


    context = compress_context(client, query, results)

    st.caption("Context compression applied")


    prompt = f"""
Answer the question using the document context.

Context:
{context}

Question:
{query}

Answer:
"""


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer using the context only."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content


    st.session_state.cache.add(query, answer)


    answer_emb = similarity_model.encode(answer, convert_to_tensor=True)
    context_emb = similarity_model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(answer_emb, context_emb).item()


    st.session_state.analytics.add_record(query, similarity)


    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer)


    st.markdown("**Confidence Score:** " + str(round(similarity, 3)))


    if similarity < 0.3:
        st.warning("⚠ Possible hallucination detected")


    with st.expander("Sources"):

        for page in sorted(sources):
            st.write("Page", page)


st.divider()

st.header("RAG Analytics")

df = st.session_state.analytics.dataframe()

if not df.empty:

    st.write("Query Confidence Scores")

    st.bar_chart(df["confidence"])

    hallucination_rate = st.session_state.analytics.hallucination_rate()

    st.metric("Hallucination Rate", f"{round(hallucination_rate * 100, 2)} %")
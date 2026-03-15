import os
from dotenv import load_dotenv
from groq import Groq

from utils.query_expansion import expand_query

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from langchain_community.retrievers import BM25Retriever

from evaluation import test_questions, evaluate_retrieval

from utils.pdf_loader import load_pdf
from utils.chunking import chunk_text
from utils.embeddings import create_vector_store


def rerank_documents(query, docs, top_k=3):

    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]


def detect_hallucination(answer, context):

    answer_embedding = similarity_model.encode(answer, convert_to_tensor=True)
    context_embedding = similarity_model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(answer_embedding, context_embedding).item()

    return similarity


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

chat_history = []


pdf_paths = [
    r"C:\Users\kruna\rag-qa-system\L1-L2-IOT- An Intro to IoT.pdf",
    r"C:\Users\kruna\rag-qa-system\NehaIOT52-57 iot paper 3.pdf"
]


all_pages = []

for path in pdf_paths:

    pages = load_pdf(path)

    all_pages.extend(pages)


if all(len(page.page_content.strip()) == 0 for page in all_pages):

    print("This PDF appears to contain no extractable text.")

    exit()


docs = chunk_text(all_pages)

vector_store = create_vector_store(docs)

bm25_retriever = BM25Retriever.from_documents(docs)

bm25_retriever.k = 10


evaluate_retrieval(test_questions, vector_store)


print("Total pages:", len(all_pages))
print("Total chunks:", len(docs))


while True:

    query = input("\nAsk a question about the PDF (type 'exit' to quit): ")

    if query.lower() == "exit":
        break


    expanded_queries = expand_query(client, query)

    results = []


    for q in expanded_queries:

        faiss_results = vector_store.similarity_search(q, k=5)

        bm25_results = bm25_retriever.invoke(q)

        results.extend(faiss_results)
        results.extend(bm25_results)


    unique_results = {}
    
    for doc in results:

        unique_results[doc.page_content] = doc

    results = list(unique_results.values())


    results = rerank_documents(query, results, top_k=3)


    if not results:

        print("No relevant information found in the document.")

        continue


    context = ""

    sources = set()


    for doc in results:

        page = doc.metadata.get("page", 0) + 1

        sources.add(page)

        context += f"(Page {page}) {doc.page_content}\n"


    history_text = ""

    for q, a in chat_history:

        history_text += f"User: {q}\nAssistant: {a}\n"


    prompt = f"""
You are a helpful assistant answering questions about a document.

Conversation History:
{history_text}

Document Context:
{context}

User Question:
{query}

Answer:
"""


    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Answer using the provided context only."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content


    except Exception as e:

        print("LLM request failed.")
        print("Reason:", str(e))

        break


    similarity_score = detect_hallucination(answer, context)


    print("\nAnswer:\n")

    print(answer)


    print("\nAnswer Confidence Score:", round(similarity_score, 3))


    if similarity_score < 0.30:

        print("⚠ Possible hallucination detected")


    print("\nSources:")

    for page in sorted(sources):

        print("Page", page)


    chat_history.append((query, answer))
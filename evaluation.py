from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

test_questions = [
    ("What is IoT?", "internet of things"),
    ("Explain IoT architecture", "iot architecture"),
    ("What are smart devices?", "smart devices"),
]

def evaluate_retrieval(test_questions, vector_store):

    correct = 0
    total = len(test_questions)

    print("\nEvaluation Results\n")
    print("{:<30} {:<10}".format("Query", "Similarity"))

    for query, expected in test_questions:

        docs = vector_store.similarity_search(query, k=3)

        context = " ".join([doc.page_content for doc in docs])

        emb1 = model.encode(context, convert_to_tensor=True)
        emb2 = model.encode(expected, convert_to_tensor=True)

        score = util.cos_sim(emb1, emb2).item()

        print("{:<30} {:.3f}".format(query, score))

        if score > 0.5:
            correct += 1

    accuracy = correct / total

    print("\nRetrieval Accuracy:", round(accuracy, 3))
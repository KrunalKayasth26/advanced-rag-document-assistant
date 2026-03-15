from sentence_transformers import SentenceTransformer, util


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.cache = []

        self.threshold = threshold


    def search(self, query):

        if not self.cache:
            return None

        query_emb = self.model.encode(query, convert_to_tensor=True)

        for item in self.cache:

            sim = util.cos_sim(query_emb, item["embedding"]).item()

            if sim > self.threshold:
                return item["answer"]

        return None


    def add(self, query, answer):

        emb = self.model.encode(query, convert_to_tensor=True)

        self.cache.append({
            "embedding": emb,
            "answer": answer
        })
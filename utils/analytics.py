import pandas as pd


class RAGAnalytics:

    def __init__(self):
        self.records = []

    def add_record(self, query, confidence):

        self.records.append({
            "query": query,
            "confidence": confidence
        })

    def dataframe(self):

        return pd.DataFrame(self.records)

    def hallucination_rate(self):

        if not self.records:
            return 0

        low = [r for r in self.records if r["confidence"] < 0.3]

        return len(low) / len(self.records)
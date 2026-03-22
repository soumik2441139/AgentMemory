from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityMatcher:
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def score(self, text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([text1, text2])
            score = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(score)
        except:
            return 0.0

    def find_best_match(self, new_text, chunks):
        best_chunk = None
        best_score = 0

        for chunk in chunks:
            if chunk.get("type") == "general":
                continue
            # Combine all chunk messages into one text
            chunk_text = " ".join([
                m["content"] for m in chunk["messages"]
            ])
            score = self.score(new_text, chunk_text)
            if score > self.threshold and score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk, best_score

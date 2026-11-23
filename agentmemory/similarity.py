from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatcher:
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def score(self, text1, text2):
        return self.scores(text1, [text2])[0]

    def scores(self, query, documents):
        """Return TF-IDF similarity scores for several documents in one fit."""
        scores = [0.0] * len(documents)
        if not query or not query.strip():
            return scores

        valid_documents = [
            (index, document)
            for index, document in enumerate(documents)
            if document and document.strip()
        ]
        if not valid_documents:
            return scores

        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(
                [query] + [document for _, document in valid_documents]
            )
            similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
            for (index, _), similarity in zip(valid_documents, similarities):
                scores[index] = float(similarity)
        except ValueError:
            # Empty vocabularies (for example, stopwords only) have no match.
            pass
        return scores

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

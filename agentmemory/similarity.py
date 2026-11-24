from collections import OrderedDict
from threading import RLock

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfIndex:
    """Reusable TF-IDF index for a fixed set of documents."""

    def __init__(self, documents, vectorizer_options):
        self.documents = tuple(documents)
        self.valid_documents = [
            (index, document)
            for index, document in enumerate(self.documents)
            if document and document.strip()
        ]
        self.vectorizer = None
        self.matrix = None

        if not self.valid_documents:
            return

        try:
            self.vectorizer = TfidfVectorizer(**vectorizer_options)
            self.matrix = self.vectorizer.fit_transform(
                [document for _, document in self.valid_documents]
            )
        except ValueError:
            self.vectorizer = None
            self.matrix = None

    def scores(self, query):
        scores = [0.0] * len(self.documents)
        if not query or not query.strip() or self.vectorizer is None:
            return scores

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix)[0]
        for (index, _), similarity in zip(self.valid_documents, similarities):
            scores[index] = float(similarity)
        return scores


class SimilarityMatcher:
    def __init__(self, threshold=0.25, max_cached_indexes=128, **vectorizer_options):
        self.threshold = threshold
        self.max_cached_indexes = max_cached_indexes
        self.vectorizer_options = vectorizer_options or {"stop_words": "english"}
        self._indexes = OrderedDict()
        self._cache_lock = RLock()

    def score(self, text1, text2):
        return self.scores(text1, [text2])[0]

    def scores(self, query, documents):
        """Score several documents with one TF-IDF fit (v1.1 behavior).

        The query is part of the fit, so query-only terms count toward its
        vector norm. This keeps the score scale that assignment thresholds
        were tuned on, and needs a single vectorizer fit per call.
        """
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
            vectorizer = TfidfVectorizer(**self.vectorizer_options)
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

    def cached_scores(self, cache_key, query, documents):
        """Reuse a bounded index and rebuild it only when candidate text changes."""
        documents = tuple(documents)
        with self._cache_lock:
            index = self._indexes.get(cache_key)
            if index is None or index.documents != documents:
                index = TfidfIndex(documents, self.vectorizer_options)
                self._indexes[cache_key] = index
            self._indexes.move_to_end(cache_key)

            while len(self._indexes) > self.max_cached_indexes:
                self._indexes.popitem(last=False)

        return index.scores(query)

    def clear_cached_index(self, cache_key=None):
        """Clear one cached session index or every cached index."""
        with self._cache_lock:
            if cache_key is None:
                self._indexes.clear()
            else:
                self._indexes.pop(cache_key, None)

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

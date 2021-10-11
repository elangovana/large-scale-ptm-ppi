import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityComparer:

    def __init__(self, n_gram=1, stop_words='english'):
        self._vectorizer = CountVectorizer(analyzer='word', ngram_range=(n_gram, n_gram), stop_words=stop_words)

    def __call__(self, list_reference, list_new):
        x_vector_ref = self._vectorizer.fit_transform(list_reference)
        y_vector_new = self._vectorizer.transform(list_new)

        sim_scores = cosine_similarity(y_vector_new, x_vector_ref, dense_output=False).toarray()

        best_score = sim_scores.max(axis=1).flatten()
        best_match_index = sim_scores.argmax(axis=1).flatten()
        best_match = np.array(list_reference)[best_match_index]

        return best_score, best_match

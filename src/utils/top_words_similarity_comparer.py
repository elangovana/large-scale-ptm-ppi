from sklearn.feature_extraction.text import CountVectorizer


class TopWordsSimilarityComparer:

    def __init__(self, n_gram=1, stop_words='english'):
        self._vectorizer = CountVectorizer(analyzer='word', ngram_range=(n_gram, n_gram), stop_words=stop_words)

    def __call__(self, list_reference, list_new):
        # Merge ref and new so no unknown words
        vec = self._vectorizer.fit(list_reference + list_new)
        word_indices = {i: w for w, i in vec.vocabulary_.items()}

        x_vector_ref = vec.transform(list_reference)
        y_vector_new = vec.transform(list_new)

        ref_word_count = x_vector_ref.sum(axis=0).tolist()[0]
        new_word_count = y_vector_new.sum(axis=0).tolist()[0]

        return ref_word_count, new_word_count, word_indices

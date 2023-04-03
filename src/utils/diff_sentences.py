import nltk
import numpy as np
from Levenshtein import editops, distance
from nltk.tokenize import word_tokenize


class DiffSentences:

    def get_edit_span(self, ref_sentence, sentence2):
        ref_words = ref_sentence.lower().split(" ")
        sent2_words = sentence2.lower().split(" ")
        diff = editops(ref_words, sent2_words)
        # The diff contains the ops [(insert, 2, 2)]
        result = []

        for (op, spos, dpos) in diff:
            if op == 'insert':
                result.append(sent2_words[dpos])
            elif op == 'replace':
                result.append(ref_words[spos])
            elif op == 'delete':
                result.append(ref_words[spos])
        return result

    def _tokenize(self, s):
        return word_tokenize(s.lower())

    def pairwise_edit_distance_ratio(self, sentences1, sentences2, score_cutoff=0.5):
        nltk.download('punkt')
        l_sentences1 = [self._tokenize(s) for s in sentences1]
        l_sentences2 = [self._tokenize(s) for s in sentences2]

        result = np.empty(shape=(len(l_sentences1), len(l_sentences2)))
        for s1i, s1 in enumerate(sentences1):
            for s2i, s2 in enumerate(sentences2):
                result[s1i, s2i] = distance(s1, s2) / len(s1)

        return result

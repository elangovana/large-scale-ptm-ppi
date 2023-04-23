import logging
import os
from multiprocessing import Pool

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

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def pairwise_edit_distance_ratio(self, sentences1, sentences2, score_cutoff=0.5):
        self._logger.info(
            f"Running pairwise_edit_distance_ratio for 2 lists of size {len(sentences1)} and  {len(sentences2)}")
        nltk.download('punkt')
        l_sentences1 = [self._tokenize(s) for s in sentences1]
        l_sentences2 = [self._tokenize(s) for s in sentences2]
        self._logger.info(f"Completed tokenization")

        result = np.empty(shape=(len(l_sentences1), len(l_sentences2)))

        parallel_params = []
        for s1i, s1 in enumerate(l_sentences1):
            for s2i, s2 in enumerate(l_sentences2):
                parallel_params.append((s1i, s2i, s1, s2))

        with Pool(os.cpu_count() - 1) as p:
            parallel_results = p.map(self._compute_edit_distance_ratio, parallel_params)

        for ((s1i, s2i, _, _), r) in zip(parallel_params, parallel_results):
            result[s1i, s2i] = r
        self._logger.info(f"Completed distance computation")

        return result

    def _compute_edit_distance_ratio(self, parallel_tuple_args):
        return distance(parallel_tuple_args[2], parallel_tuple_args[3]) / len(parallel_tuple_args[2])

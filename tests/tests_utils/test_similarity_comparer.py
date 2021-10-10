from unittest import TestCase

import numpy as np

from utils.similarity_comparer import SimilarityComparer


class TestSimilarityComparer(TestCase):

    def test__call__one(self):
        # Arrange
        train_data = [
            "This is training data",
            "This is test data"
        ]

        test_data = [
            "This is test data"
        ]

        expected_score = np.array([1.0])
        expected_match = np.array(["This is test data"])

        sut = SimilarityComparer(n_gram=1)

        # Act
        best_score, best_match = sut(train_data, test_data)

        # Assert
        self.assertSequenceEqual(expected_score, best_score)
        self.assertSequenceEqual(best_match, expected_match)

    def test__call_two(self):
        # Arrange
        train_data = [
            "Nothing is training data",
            "This is test data",
            "All rest wells"
        ]

        test_data = [
            "This is test data",
            "Nothing to do"
        ]

        expected_score = np.array([1.0, 0.5])
        expected_match = np.array(["This is test data", "Nothing is training data"])

        sut = SimilarityComparer(n_gram=1)

        # Act
        best_score, best_match = sut(train_data, test_data)

        # Assert
        self.assertSequenceEqual(expected_score.tolist(), best_score.tolist())
        self.assertSequenceEqual(best_match.tolist(), expected_match.tolist())

from unittest import TestCase

from utils.top_words_similarity_comparer import TopWordsSimilarityComparer


class TestTopWordsSimilarityComparer(TestCase):

    def test__call__one(self):
        # Arrange
        train_data = [
            "This is training data",
            "This is test data"
        ]

        test_data = [
            "This is test data"
        ]

        expected_train_word_count = {"data": 2, "training": 1, "test": 1}
        expected_test_word_count = {"data": 1, "test": 1}

        def transform_word_count(x, wi):
            return {wi[i]: c for i, c in enumerate(x) if c > 0}

        sut = TopWordsSimilarityComparer(n_gram=1)

        # Act
        actual_train_words, actual_test_words, word_indices = sut(train_data, test_data)

        # Assert
        self.assertEqual(expected_train_word_count, transform_word_count(actual_train_words, word_indices))
        self.assertEqual(expected_test_word_count, transform_word_count(actual_test_words, word_indices))

from unittest import TestCase

import numpy as np

from utils.diff_sentences import DiffSentences


class TestDiffSentences(TestCase):

    def test_get_edit_span(self):
        # Arrange
        sut = DiffSentences()

        # Act
        actual = sut.get_edit_span("This is fun", "This is not fun")

        self.assertEqual(["not"], actual)

    def test_pairwise_edit_distance_ratio(self):
        # Arrange
        sut = DiffSentences()
        expected = [[1 / 3, 0]]

        # Act
        actual = sut.pairwise_edit_distance_ratio(["This is fun"], ["This is not fun", "This is fun"])

        self.assertSequenceEqual(np.around(np.array(expected), 2).tolist(),
                                 np.around(np.array(actual), 2).tolist())

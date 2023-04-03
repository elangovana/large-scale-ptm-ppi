import io
from unittest import TestCase

from utils.diff_dataset_counterfactuals import DiffDatasetCounterFactuals


class TestDiffDatasetCounterFactuals(TestCase):

    def test_compare(self):
        # Arrange
        sut = DiffDatasetCounterFactuals()
        data = """Sentiment	Text	batch_id
Negative	Long, boring, blasphemous. Never have I been so glad to see ending credits roll.	4
Positive	Long, fascinating, soulful. Never have I been so sad to see ending credits roll.	4
"""
        expected_diff = [['boring,', 'blasphemous.', 'glad']]

        # Act
        actual = sut.compare(io.StringIO(data))

        self.assertEqual(expected_diff, actual)

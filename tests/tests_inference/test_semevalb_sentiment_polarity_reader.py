import os.path
from unittest import TestCase

from inference.semevalb_sentiment_polarity_reader import SemevalbSentimentPolarityReader


class TestSemevalbSentimentPolarityReader(TestCase):

    def test_load_dataset(self):
        # Arrange
        sample_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "semeval2017subtaskb.tsv")
        sut = SemevalbSentimentPolarityReader()

        # Act
        actual = sut.load_dataset(sample_file)

        # Assert
        self.assertEqual(11, len(actual))
        self.assertSequenceEqual(["Id", "Topic", "Sentiment", "Text"], list(actual.columns))

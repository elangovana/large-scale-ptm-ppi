import os.path
from unittest import TestCase

from inference.amazon_review_sentiment_polarity_reader import AmazonReviewSentimentPolarityReader


class TestAmazonReviewSentimentPolarityReader(TestCase):

    def test_load_dataset(self):
        # Arrange
        sample_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "amazon_polarity.csv")
        sut = AmazonReviewSentimentPolarityReader()

        # Act
        actual = sut.load_dataset(sample_file)

        self.assertEqual(27, len(actual))

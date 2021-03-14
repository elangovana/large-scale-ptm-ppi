from unittest import TestCase

from datasets.transformer_bert_tokeniser import TransformerBertTokeniser


class TestPreprocessorBertTokeniser(TestCase):

    def test_sequence_short(self):
        """
        Test case  sequences that are too short should be padded
        :return:
        """
        sut = TransformerBertTokeniser(max_feature_len=5, tokeniser=None)
        sut.item = ["THE"]
        expected = ["[CLS]", "THE", "[PAD]", "[PAD]", "[SEP]"]

        # Act
        sut.sequence_pad()

        # Assert
        self.assertSequenceEqual(expected, sut.item)

    def test_sequence_long(self):
        """
        Test case sequences that are too long should be truncated
        :return:
        """
        sut = TransformerBertTokeniser(max_feature_len=5, tokeniser=None)
        sut.item = ["THE", "dog", "ate", "a", "biscuit"]
        expected = ["[CLS]", "THE", "dog", "ate", "[SEP]"]

        # Act
        sut.sequence_pad()

        # Assert
        self.assertSequenceEqual(expected, sut.item)

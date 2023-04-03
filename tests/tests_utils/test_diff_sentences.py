from unittest import TestCase

from utils.diff_sentences import DiffSentences


class TestDiffSentences(TestCase):

    def test_call(self):
        # Arrange
        sut = DiffSentences()

        # Act
        actual = sut.get_edit_span("This is fun", "This is not fun")

        self.assertEqual(["not"], actual)

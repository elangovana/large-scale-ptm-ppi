import os
import tempfile
from unittest import TestCase

from preprocessors.kfold_aimed_json_splitter import KFoldAimedJsonSplitter


class TestKFoldAimedJsonSplitter(TestCase):

    def test_split_random(self):
        # Arrange
        data_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "aimed.json")
        k = 2
        label = "interacts"
        tempdir = tempfile.mkdtemp()

        sut = KFoldAimedJsonSplitter()

        # Act
        output_files = sut.split(data_file, tempdir, k=k, label_column=label, unique_doc_col=None)

        # Assert
        self.assertEqual(len(output_files), k)

    def test_split_unique_doc(self):
        # Arrange
        data_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "aimed.json")
        k = 2
        label = "interacts"
        tempdir = tempfile.mkdtemp()

        sut = KFoldAimedJsonSplitter()

        # Act
        output_files = sut.split(data_file, tempdir, k=k, label_column=label, unique_doc_col="documentId")

        # Assert
        self.assertEqual(len(output_files), k)

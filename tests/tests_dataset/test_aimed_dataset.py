import os
from unittest import TestCase

from datasets.aimed_dataset import AimedDataset
import pandas as pd


class TestAimedDataset(TestCase):



    def test__call__with_df_with_label(self):
        """
        Calls the dataset initialised with data frame with label
        :return:
        """
        # Arrange
        expected_y = True

        expected_x = {
            'text': 'We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL)',
            'participant1Offset': 62,
            'participant1Len': 4,
            'participant2Offset': 75,
            'participant2Len': 6,
            'otherEntities': [{'id': 'AIMed.d28.s234.e1',
                               'charOffset': 62,
                               'len': 11,
                               'text': 'GITR ligand'},
                              ],

        }
        expected = (expected_x, expected_y)

        input_payload = expected_x.copy()
        input_payload["interacts"] = expected_y
        input_df = pd.DataFrame([input_payload])

        sut = AimedDataset(input_df)

        # Act
        actual = sut[0]

        # Assert
        self.assertEqual(expected, actual)


    def test__call__with_df_no_label(self):
        """
        Calls the dataset initialised with data frame without label
        :return:
        """
        # Arrange
        # Default label when no label is provided
        expected_default_y = False

        expected_x = {
            'text': 'We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL)',
            'participant1Offset': 62,
            'participant1Len': 4,
            'participant2Offset': 75,
            'participant2Len': 6,
            'otherEntities': [{'id': 'AIMed.d28.s234.e1',
                               'charOffset': 62,
                               'len': 11,
                               'text': 'GITR ligand'},
                              ],

        }
        expected = (expected_x, expected_default_y)

        input_payload = expected_x.copy()
        input_df = pd.DataFrame([input_payload])

        sut = AimedDataset(input_df)

        # Act
        actual = sut[0]

        # Assert
        self.assertEqual(expected, actual)

    def test__call__with_file(self):
        """
        Calls the dataset initialised with file
        :return:
        """
        # Arrange
        input_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "aimed.json")
        sut = AimedDataset(input_file)

        # Act
        actual = sut[1]

        # Assert
        self.assertIsNotNone(actual)

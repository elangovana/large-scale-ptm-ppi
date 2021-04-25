import os
from unittest import TestCase

import pandas as pd

from datasets.ppi_multiclass_dataset import PpiMulticlassDataset


class TestPpiMulticlassDataset(TestCase):

    def test__call__with_df_with_label(self):
        """
        Calls the dataset initialised with data frame with label
        :return:
        """
        # Arrange
        expected_y = "phosphorylation"

        expected_x = {
            'normalised_abstract': 'Q8NHY2 (Q8NHY2) is a RING-finger-containing protein that functions to repress plant photomorphogenesis, the light-mediated programme of plant development. Mutants of Q8NHY2 are constitutively photomorphogenic, and this has been attributed to their inability to negatively regulate the proteins LAF1 (ref. 1) and HY5 (ref. 2). The role of Q8NHY2 in mammalian cells is less well characterized. Here we identify the tumour-suppressor protein P04637 as a Q8NHY2-interacting protein. Q8NHY2 increases P04637 turnover by targeting it for degradation by the proteasome in a ubiquitin-dependent fashion, independently of Q00987 or Q96PM5, which are known to interact with and negatively regulate P04637. Moreover, Q8NHY2 serves as an E3 ubiquitin ligase for P04637 in vitro and in vivo, and inhibits P04637-dependent transcription and apoptosis. Depletion of Q8NHY2 by short interfering RNA (siRNA) stabilizes P04637 and arrests cells in the G1 phase of the cell cycle. Furthermore, we identify Q8NHY2 as a P04637-inducible gene, and show that the depletion of Q8NHY2 and Q00987 by siRNA cooperatively sensitizes U2-OS cells to ionizing-radiation-induced cell death. Overall, these results indicate that Q8NHY2 is a critical negative regulator of P04637 and represents a new pathway for maintaining P04637 at low levels in unstressed cells.',
            'participant1Id': 'Q8NHY2', 'participant2Id': 'Q96PM5',
            'normalised_abstract_annotations': [{'charOffset': 0, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 8, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 166, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 338, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 439, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 451, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 479, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 496, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 613, 'len': 6, 'text': 'Q00987'},
                                                {'charOffset': 623, 'len': 6, 'text': 'Q96PM5'},
                                                {'charOffset': 688, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 706, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 750, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 792, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 851, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 902, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 987, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 999, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 1053, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 1064, 'len': 6, 'text': 'Q00987'},
                                                {'charOffset': 1196, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 1239, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 1291, 'len': 6, 'text': 'P04637'}]}
        expected = (expected_x, expected_y)

        input_payload = expected_x.copy()
        input_payload["class"] = expected_y
        input_df = pd.DataFrame([input_payload])

        sut = PpiMulticlassDataset(input_df)

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
        expected_default_y = "other"

        expected_x = {
            'normalised_abstract': 'Q8NHY2 (Q8NHY2) is a RING-finger-containing protein that functions to repress plant photomorphogenesis, the light-mediated programme of plant development. Mutants of Q8NHY2 are constitutively photomorphogenic, and this has been attributed to their inability to negatively regulate the proteins LAF1 (ref. 1) and HY5 (ref. 2). The role of Q8NHY2 in mammalian cells is less well characterized. Here we identify the tumour-suppressor protein P04637 as a Q8NHY2-interacting protein. Q8NHY2 increases P04637 turnover by targeting it for degradation by the proteasome in a ubiquitin-dependent fashion, independently of Q00987 or Q96PM5, which are known to interact with and negatively regulate P04637. Moreover, Q8NHY2 serves as an E3 ubiquitin ligase for P04637 in vitro and in vivo, and inhibits P04637-dependent transcription and apoptosis. Depletion of Q8NHY2 by short interfering RNA (siRNA) stabilizes P04637 and arrests cells in the G1 phase of the cell cycle. Furthermore, we identify Q8NHY2 as a P04637-inducible gene, and show that the depletion of Q8NHY2 and Q00987 by siRNA cooperatively sensitizes U2-OS cells to ionizing-radiation-induced cell death. Overall, these results indicate that Q8NHY2 is a critical negative regulator of P04637 and represents a new pathway for maintaining P04637 at low levels in unstressed cells.',
            'participant1Id': 'Q8NHY2', 'participant2Id': 'Q96PM5',
            'normalised_abstract_annotations': [{'charOffset': 0, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 8, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 166, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 338, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 439, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 451, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 479, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 496, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 613, 'len': 6, 'text': 'Q00987'},
                                                {'charOffset': 623, 'len': 6, 'text': 'Q96PM5'},
                                                {'charOffset': 688, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 706, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 750, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 792, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 851, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 902, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 987, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 999, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 1053, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 1064, 'len': 6, 'text': 'Q00987'},
                                                {'charOffset': 1196, 'len': 6, 'text': 'Q8NHY2'},
                                                {'charOffset': 1239, 'len': 6, 'text': 'P04637'},
                                                {'charOffset': 1291, 'len': 6, 'text': 'P04637'}]}
        expected = (expected_x, expected_default_y)

        input_payload = expected_x.copy()
        input_df = pd.DataFrame([input_payload])

        sut = PpiMulticlassDataset(input_df)

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
        input_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "train_data_ppi_multiclass",
                                  "ppi_multiclass.json")
        sut = PpiMulticlassDataset(input_file)

        # Act
        actual = sut[1]

        # Assert
        self.assertIsNotNone(actual)

    def test__call__with_dir(self):
        """
        Calls the dataset initialised with file
        :return:
        """
        # Arrange
        input_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "train_data_ppi_multiclass")
        sut = PpiMulticlassDataset(input_file)

        # Act
        actual = sut[1]

        # Assert
        self.assertIsNotNone(actual)

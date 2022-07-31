from unittest import TestCase
from unittest.mock import MagicMock

from datasets.transformer_name_normaliser import TransformerNameNormaliser


class TestTransformerNameNormaliser(TestCase):

    def test_transform(self):
        # Arrange
        sentence = "We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), and KLK3(KLK3). an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]."
        # We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), and KLK3(KLK3). an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4].
        payload = {"interacts": False
            , "text": sentence
            , "participant1Id": "AIMed.d28.s234.e1"
            , "participant1Offset": 62
            , "participant1Len": 11
            , "participant1Text": "GITR ligand"
            , "participant2Id": "AIMed.d28.s234.e2"
            , "participant2Offset": 75
            , "participant2Len": 6
            , "participant2Text": "hGITRL"
            , "documentId": "AIMed.d28"
            , "sentenceId": "AIMed.d28.s234"
            , "participantEntities": [
                {
                    "charOffset": 62
                    , "len": 11
                    , "entityType": "PROTIEN"
                },
                {
                    "charOffset": 75
                    , "len": 6
                    , "entityType": "PROTIEN"
                }
            ]
            , "otherEntities": [{"id": "AIMed.d28.s234.e0"
                                    , "charOffset": 62
                                    , "len": 4
                                    , "entityType": "PROTIEN"
                                    , "text": "GITR"},
                                {"id": "AIMed.d28.s234.e5"
                                    , "charOffset": 120
                                    , "len": 4
                                    , "entityType": "PROTIEN"
                                    , "text": "KLK3"},
                                {"id": "AIMed.d28.s234.e6"
                                    , "charOffset": 125
                                    , "len": 4
                                    , "entityType": "PROTIEN"
                                    , "text": "KLK3"}
                                ]
                   }

        mock_name_replacer = MagicMock()
        sut = TransformerNameNormaliser(text_key="text", participants_entities_dict_key="participantEntities",
                                        other_entities_dict_key="otherEntities", name_replacer=mock_name_replacer,
                                        random_seed=30)

        # Act
        sut(payload)

        # Assert
        mock_name_replacer.assert_called_with(
            entities=[{'charOffset': 62, 'len': 4, 'replacement': 'PROTIG1', "original": "GITR"},
                      {'charOffset': 120, 'len': 4, 'replacement': 'PROTIG0', "original": "KLK3"},
                      {'charOffset': 125, 'len': 4, 'replacement': 'PROTIG0', "original": "KLK3"},
                      {'charOffset': 62, 'len': 11, 'replacement': 'PROTPART0', "original": "GITR ligand"},
                      {'charOffset': 75, 'len': 6, 'replacement': 'PROTPART1', "original": "hGITRL"},
                      ],
            text=sentence)

from unittest import TestCase

from datasets.transformer_aimed_participant_augmentor import TransformerAimedParticipantAugmentor


class TestTransformerAimedParticipantAugmentor(TestCase):

    def test__call__(self):
        # Arrange

        payload = {"interacts": True
            , "text": "test sentence"
            , "participant1Id": "AIMed.d28.s234.e0"
            , "participant1Offset": 62
            , "participant1Len": 4
            , "participant1Text": "GITR"
            , "participant2Id": "AIMed.d28.s234.e1"
            , "participant2Offset": 62
            , "participant2Len": 11
            , "participant2Text": "GITR ligand"
            , "documentId": "AIMed.d28"
            , "sentenceId": "AIMed.d28.s234"
            , "otherEntities": [{"id": "AIMed.d28.s234.e2"
                                    , "charOffset": 75
                                    , "len": 6
                                    , "text": "hGITRL"}
                                ]
                   }

        expected = payload.copy()
        expected["participantEntities"] = [
            {"id": "AIMed.d28.s234.e0"
                , "charOffset": 62
                , "len": 4
                , "text": "GITR"
             },
            {"id": "AIMed.d28.s234.e1"
                , "charOffset": 62
                , "len": 11
                , "text": "GITR ligand"
             }
        ]

        sut = TransformerAimedParticipantAugmentor(participant1_id_key="participant1Id"
                                                   , participant1_offset_key="participant1Offset"
                                                   , participant1_len_key="participant1Len"
                                                   , participant1_text_key="participant1Text"
                                                   , participant2_id_key="participant2Id"
                                                   , participant2_offset_key="participant2Offset"
                                                   , participant2_len_key="participant2Len"
                                                   , participant2_text_key="participant2Text"
                                                   , result_key="participantEntities")

        # Act
        actual = sut(payload)

        # Assert
        self.assertEqual(expected, actual)

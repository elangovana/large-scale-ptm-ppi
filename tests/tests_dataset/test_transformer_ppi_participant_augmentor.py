from unittest import TestCase

from datasets.transformer_ppi_participant_augmentor import TransformerPPIParticipantAugmentor


class TestTransformerPPIParticipantAugmentor(TestCase):

    def test__call__(self):
        # Arrange

        payload = {"interacts": True
            , "normalised_abstract_annotations": [{"charOffset": 0, "len": 6, "text": "P30291"},
                                                  {"charOffset": 12, "len": 6, "text": "P06493"},
                                                  {"charOffset": 119, "len": 6, "text": "P06493"},
                                                  {"charOffset": 172, "len": 6, "text": "P30291"},
                                                  {"charOffset": 180, "len": 6, "text": "P30291"},
                                                  {"charOffset": 355, "len": 5, "text": "Q9UKB"}]
            , "participant1Id": "P30291"
            , "participant2Id": "P06493"
            , "normalised_abstract": "P30291, the P06493 inhibitory kinase, ....."
                   }

        expected = payload.copy()
        DEFAULT_ENTITY_TYPE = "PROT"
        expected["participantEntities"] = [
            {"charOffset": 0
                , "len": 6
                , "entityType": DEFAULT_ENTITY_TYPE

             },
            {"charOffset": 12
                , "len": 6
                , "entityType": DEFAULT_ENTITY_TYPE
             },
            {"charOffset": 119
                , "len": 6
                , "entityType": DEFAULT_ENTITY_TYPE
             },
            {"charOffset": 172
                , "len": 6
                , "entityType": DEFAULT_ENTITY_TYPE

             },
            {"charOffset": 180
                , "len": 6
                , "entityType": DEFAULT_ENTITY_TYPE

             }
        ]
        expected["otherEntities"] = [

            {"charOffset": 355
                , "len": 5
                , "entityType": DEFAULT_ENTITY_TYPE
             }
        ]

        sut = TransformerPPIParticipantAugmentor(participant1_key="participant1Id"
                                                 , participant2_key="participant2Id"
                                                 , annotations_dict_key="normalised_abstract_annotations"
                                                 , result_key_participant="participantEntities"
                                                 , result_key_other="otherEntities",
                                                 annotation_entity_type_default=DEFAULT_ENTITY_TYPE

                                                 )

        # Act
        actual = sut(payload)

        # Assert
        self.assertEqual(expected, actual)

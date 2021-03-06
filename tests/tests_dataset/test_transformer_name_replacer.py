from unittest import TestCase

from datasets.transformer_name_replacer import TransformerNameReplacer


class TestTransformerNameReplacer(TestCase):

    def test_transform(self):
        # Arrange
        raw_text = "This is KLK3"
        entities = [{"charOffset": 8
                        , "len": 4
                        , "replacement": "Protein1"
                     }]

        expected_text = "This is Protein1"
        sut = TransformerNameReplacer()

        # Act
        actual = sut(text=raw_text, entities=entities)

        # Assert
        self.assertEqual(expected_text, actual)


    def test_transform_two_replacements(self):
        # Arrange
        raw_text = "This is KLK3 and KLK4"
        entities = [{"charOffset": 17
                        , "len": 4
                        , "replacement": "Protein2"
                     },
                    {"charOffset": 8
                        , "len": 4
                        , "replacement": "Protein1"
                     }]

        expected_text = "This is Protein1 and Protein2"
        sut = TransformerNameReplacer()

        # Act
        actual = sut(text=raw_text, entities=entities)

        # Assert
        self.assertEqual(expected_text, actual)

    def test_transform_two_replacements_startposition(self):
        # Arrange
        raw_text = "KLK3 and KLK4"
        entities = [{"charOffset": 0
                        , "len": 4
                        , "replacement": "Protein1"
                     },
                    {"charOffset": 9
                        , "len": 4
                        , "replacement": "Protein2"
                     }]

        expected_text = "Protein1 and Protein2"
        sut = TransformerNameReplacer()

        # Act
        actual = sut(text=raw_text, entities=entities)

        # Assert
        self.assertEqual(expected_text, actual)

    def test_transform_overlap_replacements(self):
        # Arrange
        raw_text = "This is KLK3 and KLK4 and klk5"

        entities = [{"charOffset": 9
                        , "len": 4
                        , "replacement": "Protein2"
                     },
                    {"charOffset": 8
                        , "len": 4
                        , "replacement": "Protein1"
                     },
                    {"charOffset": 26
                        , "len": 4
                        , "replacement": "Protein3"
                     }
                    ]

        expected_text = "This is Protein1 and KLK4 and Protein3"
        sut = TransformerNameReplacer()

        # Act
        actual = sut(text=raw_text, entities=entities)

        # Assert
        self.assertEqual(expected_text, actual)


    def test_transform_three_replacements(self):
        # Arrange
        raw_text = "This is KLK3 and KLK4s and testlabs and magic"
        entities = [{"charOffset": 17
                        , "len": 5
                        , "replacement": "Protein2"
                     },
                    {"charOffset": 8
                        , "len": 4
                        , "replacement": "Protein1"
                     },
                    {"charOffset": 27
                        , "len": 8
                        , "replacement": "Protein3"
                     }

                    ]

        expected_text = "This is Protein1 and Protein2 and Protein3 and magic"
        sut = TransformerNameReplacer()

        # Act
        actual = sut(text=raw_text, entities=entities)

        # Assert
        self.assertEqual(expected_text, actual)


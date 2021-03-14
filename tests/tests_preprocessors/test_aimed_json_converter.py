import os
import tempfile
from unittest import TestCase

from preprocessors.aimed_json_converter import AIMedJsonConverter


class TestAIMedJsonConverter(TestCase):

    def test_convert_fromstring_single_document_no_entities(self):
        """
        Test case document with no entities
        :return:
        """
        # Arrange
        sentence = "Identification of a new member of the tumor necrosis factor family and its receptor, a human ortholog of mouse GITR."
        data = """<corpus source="AIMed">
            <document id="AIMed.d27">
                <sentence id="AIMed.d27.s232" text="%s" seqId="s232">
                </sentence>
            </document>
            </corpus>
        """ % sentence

        expected_json = []

        sut = AIMedJsonConverter()

        # Act
        actual = sut.convert_fromstring(data)

        # Assert
        self.assertEqual(expected_json, actual)

    def test_convert_fromstring_single_document_with_entities_and_interactions(self):
        """
        Test case document with  entities and interactions
        :return:
        """
        # Arrange
        sentence = "We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]."
        data = """<corpus source="AIMed">
            <document id="AIMed.d28">
                <sentence id="AIMed.d28.s234" text="%s" seqId="s234">
                  <entity id="AIMed.d28.s234.e0" charOffset="62-65" type="protein" text="GITR" seqId="e329"/>
                  <entity id="AIMed.d28.s234.e1" charOffset="62-72" type="protein" text="GITR ligand" seqId="e330"/>
                  <interaction id="AIMed.d28.s234.i0" e1="AIMed.d28.s234.e0" e2="AIMed.d28.s234.e1" type="None" directed="false" seqId="i0"/>
                 
                </sentence>
            </document>
            </corpus>
        """ % sentence

        expected_json = [
            {"interacts": True
                , "text" :sentence
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
                , "otherEntities": []
             }
        ]

        sut = AIMedJsonConverter()

        # Act
        actual = sut.convert_fromstring(data)

        # Assert
        self.assertEqual(expected_json, actual)

    def test_convert_fromstring_single_document_with_entities_and_interactions_with_non_participants(self):
        """
        Test case document with  entities and interactions with non_participating entities
        :return:
        """
        # Arrange
        sentence = "We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]."
        data = """<corpus source="AIMed">
            <document id="AIMed.d28">
                <sentence id="AIMed.d28.s234" text="%s" seqId="s234">
                  <entity id="AIMed.d28.s234.e0" charOffset="62-65" type="protein" text="GITR" seqId="e329"/>
                  <entity id="AIMed.d28.s234.e1" charOffset="62-72" type="protein" text="GITR ligand" seqId="e330"/>
                  <entity id="AIMed.d28.s234.e2" charOffset="75-80" type="protein" text="hGITRL" seqId="e332"/>
                  <interaction id="AIMed.d28.s234.i0" e1="AIMed.d28.s234.e0" e2="AIMed.d28.s234.e1" type="None" directed="false" seqId="i0"/>

                </sentence>
            </document>
            </corpus>
        """ % sentence

        expected_json = [
            {"interacts": True
                , "text": sentence
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
             },
            {"interacts": False
                , "text": sentence
                , "participant1Id": "AIMed.d28.s234.e0"
                , "participant1Offset": 62
                , "participant1Len": 4
                , "participant1Text": "GITR"
                , "participant2Id": "AIMed.d28.s234.e2"
                , "participant2Offset": 75
                , "participant2Len": 6
                , "participant2Text": "hGITRL"
                , "documentId": "AIMed.d28"
                , "sentenceId": "AIMed.d28.s234"
                , "otherEntities": [{"id": "AIMed.d28.s234.e1"
                                        , "charOffset": 62
                                        , "len": 11
                                        , "text": "GITR ligand"}
                                    ]
             }
            , {"interacts": False
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
                , "otherEntities": [{"id": "AIMed.d28.s234.e0"
                                        , "charOffset": 62
                                        , "len": 4
                                        , "text": "GITR"}
                                    ]
               }

        ]

        sut = AIMedJsonConverter()

        # Act
        actual = sut.convert_fromstring(data)

        # Assert
        self.assertEqual(expected_json, actual)

    def test_convert_fromstring_multiple_sentences(self):
        """
        Test case with multiple sentences
        :return:
        """
        # Arrange
        sentence_44 = "This work shows that single and double Ala substitutions of His18 and Phe21 in IL-8 reduced up to 77-fold the binding affinity to IL-8 receptor subtypes A (CXCR1) and B (CXCR2) and to the Duffy antigen."
        sentence_34 = "We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]."
        data = """<corpus source="AIMed">
            <document id="AIMed.d29">
                <sentence id="AIMed.d29.s244" text="%s" seqId="s244">
                  <entity id="AIMed.d29.s244.e1" charOffset="130-133" type="protein" text="IL-8" seqId="e364"/>
                  <entity id="AIMed.d29.s244.e5" charOffset="188-200" type="protein" text="Duffy antigen" seqId="e373"/>
                  <interaction id="AIMed.d29.s244.i0" e1="AIMed.d29.s244.e1" e2="AIMed.d29.s244.e5" type="None" directed="false" seqId="i3"/>
                </sentence>
                <sentence id="AIMed.d29.s234" text="%s" seqId="s234">
                  <entity id="AIMed.d29.s234.e0" charOffset="62-65" type="protein" text="GITR" seqId="e329"/>
                  <entity id="AIMed.d29.s234.e1" charOffset="62-72" type="protein" text="GITR ligand" seqId="e330"/>
                  <interaction id="AIMed.d29.s234.i0" e1="AIMed.d29.s234.e0" e2="AIMed.d29.s234.e1" type="None" directed="false" seqId="i0"/>
                </sentence>
            </document>
            </corpus>
        """ % (sentence_44, sentence_34)

        expected_json = [
            {"interacts": True
             , "text":sentence_44
                , "participant1Id": "AIMed.d29.s244.e1"
                , "participant1Offset": 130
                , "participant1Len": 4
                , "participant1Text": "IL-8"
                , "participant2Id": "AIMed.d29.s244.e5"
                , "participant2Offset": 188
                , "participant2Len": 13
                , "participant2Text": "Duffy antigen"
                , "documentId": "AIMed.d29"
                , "sentenceId": "AIMed.d29.s244"
                , "otherEntities": []
             },

            {"interacts": True
                , "text": sentence_34
                , "participant1Id": "AIMed.d29.s234.e0"
                , "participant1Offset": 62
                , "participant1Len": 4
                , "participant1Text": "GITR"
                , "participant2Id": "AIMed.d29.s234.e1"
                , "participant2Offset": 62
                , "participant2Len": 11
                , "participant2Text": "GITR ligand"
                , "documentId": "AIMed.d29"
                , "sentenceId": "AIMed.d29.s234"
                , "otherEntities": []
             }
        ]

        sut = AIMedJsonConverter()

        # Act
        actual = sut.convert_fromstring(data)

        # Assert
        self.assertEqual(expected_json, actual)

    def test_convert_fromstring_multiple_documents(self):
        """
        Test case with multiple documents
        :return:
        """
        # Arrange

        sentence_44 = "This work shows that single and double Ala substitutions of His18 and Phe21 in IL-8 reduced up to 77-fold the binding affinity to IL-8 receptor subtypes A (CXCR1) and B (CXCR2) and to the Duffy antigen."
        sentence_34 = "We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]."


        data = """<corpus source="AIMed">
            <document id="AIMed.d29">
                <sentence id="AIMed.d29.s244" text="%s" seqId="s244">
                  <entity id="AIMed.d29.s244.e1" charOffset="130-133" type="protein" text="IL-8" seqId="e364"/>
                  <entity id="AIMed.d29.s244.e5" charOffset="188-200" type="protein" text="Duffy antigen" seqId="e373"/>
                  <interaction id="AIMed.d29.s244.i0" e1="AIMed.d29.s244.e1" e2="AIMed.d29.s244.e5" type="None" directed="false" seqId="i3"/>
                </sentence>
            </document>
            <document id="AIMed.d28">
                <sentence id="AIMed.d28.s234" text="%s" seqId="s234">
                  <entity id="AIMed.d28.s234.e0" charOffset="62-65" type="protein" text="GITR" seqId="e329"/>
                  <entity id="AIMed.d28.s234.e1" charOffset="62-72" type="protein" text="GITR ligand" seqId="e330"/>
                  <interaction id="AIMed.d28.s234.i0" e1="AIMed.d28.s234.e0" e2="AIMed.d28.s234.e1" type="None" directed="false" seqId="i0"/>
                </sentence>
            </document>
            </corpus>
        """ % (sentence_44, sentence_34)

        expected_json = [
            {"interacts": True
                , "text": sentence_44
                , "participant1Id": "AIMed.d29.s244.e1"
                , "participant1Offset": 130
                , "participant1Len": 4
                , "participant1Text": "IL-8"
                , "participant2Id": "AIMed.d29.s244.e5"
                , "participant2Offset": 188
                , "participant2Len": 13
                , "participant2Text": "Duffy antigen"
                , "documentId": "AIMed.d29"
                , "sentenceId": "AIMed.d29.s244"
                , "otherEntities": []
             },

            {"interacts": True
                , "text": sentence_34
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
                , "otherEntities": []
             }
        ]

        sut = AIMedJsonConverter()

        # Act
        actual = sut.convert_fromstring(data)

        # Assert
        self.assertEqual(expected_json, actual)

    def test_convert_fromfile(self):
        # Arrange
        input_xml_file = os.path.join(os.path.dirname(__file__), "..", "sample_data", "aimed.xml")
        sut = AIMedJsonConverter()
        dest_json_file = os.path.join( tempfile.mkdtemp(), "sample.json")

        # Act
        sut.convert_fromfile(input_xml_file,  dest_json_file)

        self.assertTrue(os.path.isfile(dest_json_file ))


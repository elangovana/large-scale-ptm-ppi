import argparse
import itertools
import json
import logging
import sys
import xml.etree.ElementTree as ET


class AIMedJsonConverter:
    """
    Converts the Pyysalo formatted XML to flattened json ( genereted by convert_aimed.xml here http://mars.cs.utu.fi/PPICorpora/)
    """

    def convert_fromstring(self, xml_string):
        """
        Flattens xml file into Json with positive and negative interactions
        :param xml_string: The xml file or handle. The XML file is formatted as follows
        <corpus source="AIMed">
            <document id="AIMed.d27">
                <sentence id="AIMed.d27.s232" text="Identification of a new member of the tumor necrosis factor family and its receptor, a human ortholog of mouse GITR." seqId="s232">
                </sentence>
            <document>
            <document id="AIMed.d28">
                <sentence id="AIMed.d28.s232" text="Identification of a new member of the tumor necrosis factor family and its receptor, a human ortholog of mouse GITR." seqId="s232">
                </sentence>
                <sentence id="AIMed.d28.s234" text="We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]." seqId="s234">
                  <entity id="AIMed.d28.s234.e0" charOffset="62-65" type="protein" text="GITR" seqId="e329"/>
                  <entity id="AIMed.d28.s234.e1" charOffset="62-72" type="protein" text="GITR ligand" seqId="e330"/>
                  <entity id="AIMed.d28.s234.e5" charOffset="162-212" type="protein" text="glucocorticoid-induced TNFR-related (mGITR) protein" seqId="e339"/>
                  <interaction id="AIMed.d28.s234.i0" e1="AIMed.d28.s234.e0" e2="AIMed.d28.s234.e2" type="None" directed="false" seqId="i0"/>
                  <interaction id="AIMed.d28.s234.i1" e1="AIMed.d28.s234.e1" e2="AIMed.d28.s234.e3" type="None" directed="false" seqId="i1"/>
                  <interaction id="AIMed.d28.s234.i2" e1="AIMed.d28.s234.e2" e2="AIMed.d28.s234.e3" type="None" directed="false" seqId="i2"/>
                </sentence>
            </document>
        </corpus>
        :return:
        """
        root = ET.fromstring(xml_string)

        return self._parse_xml(root)

    def convert_fromfile(self, src_xml_file, dest_json_file):
        """
Converts from source file
        :param src_xml_file: Source input xml file
        :param dest_json_file: Destination output json file
        """
        tree = ET.parse(src_xml_file)
        results = self._parse_xml(tree.getroot())

        with open(dest_json_file, "w") as f:
            f.write(json.dumps(results))

    def _parse_xml(self, root):
        result = []
        for document in root.findall('document'):
            items = self._parse_document(document)
            result.extend(items)
        return result

    def _parse_document(self, document):
        doc_id = document.attrib["id"]
        result = []
        for sentence in document.findall("sentence"):
            items = self._parse_sentence(sentence, doc_id)
            result.extend(items)
        return result

    def _parse_sentence(self, sentence, doc_id):
        sentence_id = sentence.attrib["id"]
        sentence_text= sentence.attrib["text"]
        all_entities = []
        all_interactions = []
        # get entities
        for entity in sentence.findall("entity"):
            assert entity.attrib["type"] == "protein"
            e_start, e_end = entity.attrib["charOffset"].split("-")

            all_entities.append({
                "id": entity.attrib["id"]
                , "charOffset": int(e_start)
                , "len": int(e_end) - int(e_start) + 1
                , "text": entity.attrib["text"]
            })

        # get interactions
        for interaction in sentence.findall("interaction"):
            all_interactions.append({
                "id": interaction.attrib["id"]
                , "e1": interaction.attrib["e1"]
                , "e2": interaction.attrib["e2"]
            })

        # generate interactions
        return self._generate_records(all_entities, all_interactions,sentence_text, sentence_id, doc_id)

    def _generate_records(self, entities, interactions,sentence, sentence_id, doc_id):
        result = []
        # Get interactions as set
        interactions_set = []
        for interaction in interactions:
            interactions_set.append(frozenset([interaction["e1"], interaction["e2"]]))
        interactions_set = set(interactions_set)

        # generate positive and negative using combinations
        for e in itertools.combinations(entities, 2):
            # sort entities by id, so that e1 is always first
            e1, e2 = sorted(e, key=lambda x: x["id"])

            entities_pair_set = frozenset([e1["id"], e2["id"]])
            is_positive = entities_pair_set in interactions_set

            # Rest of the entities except the ones participating
            other_entities = [e for e in entities if e["id"] not in entities_pair_set]

            result.append({"interacts": is_positive
                              , "text":sentence
                              , "participant1Id": e1["id"]
                              , "participant1Offset": e1["charOffset"]
                              , "participant1Len": e1["len"]
                              , "participant1Text": e1["text"]
                              , "participant2Id": e2["id"]
                              , "participant2Offset": e2["charOffset"]
                              , "participant2Len": e2["len"]
                              , "participant2Text": e2["text"]
                              , "documentId": doc_id
                              , "sentenceId": sentence_id
                              , "otherEntities": other_entities
                           })

        return result


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfile",
                        help="The input XML Aimed file", required=True)

    parser.add_argument("--outputfile",
                        help="The output json file", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    AIMedJsonConverter().convert_fromfile(args.inputfile, args.outputfile)


if __name__ == "__main__":
    run_main()

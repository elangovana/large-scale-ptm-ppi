import argparse
import csv
import itertools
import json
import logging
import sys


class ChemprotAbstractJsonConverter:
    """
    Converts chem prot into json https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/, with abstract intact
    """

    def convert(self, abstract_file, ner_file, dest_json_file, relationship_file=None):
        result = []
        abstract_dict = self._get_abstracts_dict(abstract_file)
        entities_dict = self._get_entities_dict(ner_file)
        if relationship_file:
            relations_dict = self._get_relationship(relationship_file)
        else:
            relations_dict = self._get_relationship_from_entities(entities_dict)
        for abstract_id, relations in relations_dict.items():
            entities_in_relationship = []

            relations_found_in_file = []

            for r in relations:
                p1_id = r["p1"]
                p2_id = r["p2"]
                relations_found_in_file.append(frozenset([entities_dict[abstract_id][p1_id]["entity_name"],
                                                          entities_dict[abstract_id][p2_id]["entity_name"]]))

                p1 = entities_dict[abstract_id][p1_id]
                p2 = entities_dict[abstract_id][p2_id]
                # {
                #     "charOffset": 0,
                #     "len": 6,
                #     "text": "P30291"
                # },
                entities_in_relationship.append(frozenset([p1_id, p2_id]))
                label = r["relationship_group"] if r["is_eval"] == "Y" else "NEGATIVE"
                result.append({
                    "abstract_id": abstract_id,
                    "abstract": abstract_dict[abstract_id],
                    "participant1_id": p1_id,
                    "participant1": p1,
                    "participant2_id": p2_id,
                    "participant2": p2,
                    "annotations": list(entities_dict[abstract_id].values()),
                    "relationship_type": r["relationship_type"],
                    "relationship_group": r["relationship_group"],
                    "is_eval": r["is_eval"],
                    "label": label
                })

        if dest_json_file:
            with open(dest_json_file, "w") as f:
                json.dump(result, f)

        return result

    def _get_abstracts_dict(self, abstract_file_or_handle):
        if isinstance(abstract_file_or_handle, str):
            with open(abstract_file_or_handle, "r") as a:
                abstracts = self._parse_abstracts(a)
        else:
            abstracts = self._parse_abstracts(abstract_file_or_handle)
        return abstracts

    def _parse_abstracts(self, a):
        abstracts = {}
        reader = csv.reader(a, delimiter='\t')
        for row in reader:
            id = row[0]
            # The rows may have abstract and title or just the abstract
            text = " ".join(row[1:])
            abstracts[id] = text
        return abstracts

    def _get_entities_dict(self, ner_file_or_handle):
        if isinstance(ner_file_or_handle, str):
            with open(ner_file_or_handle, "r") as a:
                ners = self._parse_entities(a)
        else:
            ners = self._parse_entities(ner_file_or_handle)
        return ners

    def _parse_entities(self, a):
        ners = {}
        reader = csv.reader(a, delimiter='\t', quotechar=None)
        for row in reader:
            abstract_id = row[0]
            id = row[1]
            entity_type = row[2]
            start_pos = int(row[3])
            end_pos = int(row[4])
            entity_name = row[5]
            if abstract_id not in ners: ners[abstract_id] = {}
            ners[abstract_id][id] = {
                "abstract_id": abstract_id,
                "id": id,
                "entity_type": entity_type,
                "charOffset": start_pos,
                "len": end_pos - start_pos,
                "entity_name": entity_name,
                "text": id
            }
        return ners

    def _get_relationship(self, rel_file_or_handler):
        if isinstance(rel_file_or_handler, str):
            with open(rel_file_or_handler, "r") as a:
                rels = self._parse_relationship(a)
        else:
            rels = self._parse_relationship(rel_file_or_handler)
        return rels

    def _parse_relationship(self, a):
        rels = {}
        reader = csv.reader(a, delimiter='\t')
        for row in reader:
            abstract_id = row[0]
            group = row[1]
            is_eval = row[2]
            rel_type = row[3]
            p1 = row[4].replace("Arg1:", "")
            p2 = row[5].replace("Arg2:", "")
            if not abstract_id in rels: rels[abstract_id] = []

            rels[abstract_id].append({
                "relationship_group": group,
                "is_eval": is_eval.strip(" "),
                "relationship_type": rel_type,
                "p1": p1,
                "p2": p2

            })
        return rels

    def _get_relationship_from_entities(self, entities_dict):
        rels = {}
        for abstract_id, abstract_entity_dict in entities_dict.items():
            if abstract_id not in rels:
                rels[abstract_id] = []

            # Relationship only exist between gene and protein, so we only need to generate those tuples
            genes = []
            chemicals = []
            for entity_id, entity_details in abstract_entity_dict.items():
                if entity_details["entity_type"] in ["GENE", "GENE-Y", "GENE-N"]:
                    genes.append(entity_id)
                elif entity_details["entity_type"] == "CHEMICAL":
                    chemicals.append(entity_id)

            for g, c in itertools.product(genes, chemicals):
                rels[abstract_id].append({
                    "relationship_group": None,
                    "is_eval": "Y",
                    "relationship_type": "NEGATIVE",
                    "p1": g,
                    "p2": c

                })
        return rels


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--abstractfile",
                        help="The input abstract file", required=True)
    parser.add_argument("--entitiesfile",
                        help="The input entities file", required=True)
    parser.add_argument("--relfile",
                        help="The input relationships file", required=False, default=None)

    parser.add_argument("--outputfile",
                        help="The output json file", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    ChemprotAbstractJsonConverter().convert(args.abstractfile, args.entitiesfile, args.outputfile, args.relfile)


if __name__ == "__main__":
    run_main()

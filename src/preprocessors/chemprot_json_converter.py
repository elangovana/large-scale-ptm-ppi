import argparse
import csv
import itertools
import json
import logging
import sys


class ChemprotJsonConverter:
    """
    Converts chem prot into json https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/
    """

    def convert(self, abstract_file, ner_file, relationship_file, dest_json_file):
        result = []
        abstract_dict = self._get_abstracts_dict(abstract_file)
        entities_dict = self._get_entities_dict(ner_file)
        relations_dict = self._get_relationship(relationship_file)
        for abstract_id, relations in relations_dict.items():
            entities_in_relationship = []
            for r in relations:
                p1_id = r["p1"]
                p2_id = r["p2"]
                s, p1, p2 = self._extract_sentence(abstract_dict[abstract_id],
                                                   entities_dict[abstract_id][p1_id],
                                                   entities_dict[abstract_id][p2_id])
                entities_in_relationship.append(frozenset([p1_id, p2_id]))
                result.append({
                    "abstract_id": abstract_id,
                    "abstract": abstract_dict[abstract_id],
                    "sentence": s,
                    "participant1_id": p1["id"],
                    "participant1": p1,
                    "participant2_id": p2["id"],
                    "participant2": p2,
                    "relationship_type": r["relationship_type"],
                    "relationship_group": r["relationship_group"],
                    "is_eval": r["is_eval"]
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
            title = row[1]
            text = title + " " + row[2]
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
        reader = csv.reader(a, delimiter='\t')
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
                "start_pos": start_pos,
                "end_pos": end_pos,
                "entity_name": entity_name
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

    def _generate_records(self, entities, interactions, sentence, sentence_id, doc_id):
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
                              , "text": sentence
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

    def _extract_sentence(self, abstract, p1_details, p2_details):
        """
        Use preprocessing as mentioned in https://academic.oup.com/database/article/doi/10.1093/database/bay060/5042822

        Pre-processing involves sentence splitting and anonymizing target entities and chemical compounds.
        Abstract data usually consists of several sentences.
        However, we found that almost all the gold-standard relations exist between two target entities in the same sentence.
        Therefore, we split an abstract into sentences and assumed that a sentence had a candidate relationship if it contained at least two entities.
        We used only the sentences with candidate relationships and ignored the other sentences.
        Among the candidate relationships, we labeled the gold-standard relations as true instances and the others as negative instances.
        """

        # Ensure p1 is occurs before p2
        p1_details = p1_details.copy()
        p2_details = p2_details.copy()
        if p1_details["start_pos"] > p2_details["start_pos"]:
            t = p1_details
            p1_details = p2_details
            p2_details = t

        s = p1_details["start_pos"]
        e = p2_details["end_pos"]

        # This is a loose heuristic for sentence split, there are "." in the abstracts that are not EOS.
        sentence_start = abstract[:s].rfind(". ")
        if sentence_start < 0:
            sentence_start = 0
        sentence_start += 2

        sentence_end = abstract[e:].find(". ")
        if sentence_end < 0:
            sentence_end = len(abstract) - 1
        else:
            sentence_end = e + sentence_end

        p1_details["start_pos"] = p1_details["start_pos"] - sentence_start
        p1_details["end_pos"] = p1_details["end_pos"] - sentence_start
        p2_details["start_pos"] = p2_details["start_pos"] - sentence_start
        p2_details["end_pos"] = p2_details["end_pos"] - sentence_start

        sentence = abstract[sentence_start:sentence_end]
        return sentence, p1_details, p2_details


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--abstractfile",
                        help="The input abstract file", required=True)
    parser.add_argument("--entitiesfile",
                        help="The input entities file", required=True)
    parser.add_argument("--relfile",
                        help="The input relationships file", required=True)

    parser.add_argument("--outputfile",
                        help="The output json file", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    ChemprotJsonConverter().convert(args.abstractfile, args.entitiesfile, args.relfile, args.outputfile)


if __name__ == "__main__":
    run_main()

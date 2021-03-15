from random import Random

from datasets.transformer_name_replacer import TransformerNameReplacer


class TransformerNameNormaliser:

    def __init__(self, text_key: str, participant1_offset_key: str, participant1_len_key: str,
                 participant2_offset_key: str, participant2_len_key: str, other_entities_dict_key: str,
                 name_replacer=None, random_seed=None):
        """
        Normalise entities to a predefined set
        :param text_key: The name of the text field key
        :param participant1_offset_key: The participant 1 offset key
        :param participant1_len_key: The participant 1 len  key
        :param participant2_offset_key: The participant 2 offset  key
        :param participant2_len_key: The participant 2 len  key
        :param other_entities_dict_key: Other entities key, and this is a dict with at least these 2 keys
                {"charOffset": 62
                , "len": 4 }
        :param name_replacer: An optional transformer to perform the replacement
        :param random_seed: Use random seed to fix the protein name replacements

        """
        self.text_key = text_key
        # Participant 1
        self.participant1_offset_key = participant1_offset_key
        self.participant1_len_key = participant1_len_key
        # Participant 2
        self.participant2_offset_key = participant2_offset_key
        self.participant2_len_key = participant2_len_key
        # Other entities
        self.other_entities_dict_key = other_entities_dict_key

        self.name_replacer = name_replacer or TransformerNameReplacer()

        self._participants_fmt = ["PROTEINA", "PROTEINB"]
        self._other_proteins_fmt = "PROTEIN{}"

        self._random = Random(random_seed)

    def __call__(self, payload):
        other_entities = payload[self.other_entities_dict_key]
        raw_text = payload[self.text_key]

        # Get replacement for non-participants
        unique_other_entities = set(
            [raw_text[e["charOffset"]:(e["charOffset"] + 1 + e["len"])] for e in other_entities])

        randomised_list = self._random.sample(list(unique_other_entities), k=len(unique_other_entities))
        random_replacement = {v: self._other_proteins_fmt.format(i) for i, v in enumerate(randomised_list)}

        entities_replacements = []
        for e in other_entities:
            s_pos = e["charOffset"]
            e_pos = s_pos + 1 + e["len"]
            replacement_text = random_replacement[raw_text[s_pos: e_pos]]
            entities_replacements.append({
                "charOffset": e["charOffset"]
                , "len": e["len"]
                , "replacement": replacement_text
            })
        print(entities_replacements)

        p1_start = payload[self.participant1_offset_key]
        p1_len = payload[self.participant1_len_key]
        p1_end = p1_start + p1_len + 1
        participant1 = raw_text[p1_start: p1_end]

        p2_start = payload[self.participant2_offset_key]
        p2_len = payload[self.participant2_len_key]
        p2_end = p2_start + p2_len + 1
        participant2 = raw_text[p2_start: p2_end]

        # Map participants to replacement
        random_participant_replacement = {[participant1, participant2][i]: v for i, v in
                                          enumerate(self._random.sample(self._participants_fmt,
                                                                        k=len(self._participants_fmt)))}

        entities_replacements.append({
            "charOffset": p1_start
            , "len": p1_len
            , "replacement": random_participant_replacement[participant1]
        })

        entities_replacements.append({
            "charOffset": p2_start
            , "len": p2_len
            , "replacement": random_participant_replacement[participant2]
        })

        return self.name_replacer(text=raw_text, entities=entities_replacements)

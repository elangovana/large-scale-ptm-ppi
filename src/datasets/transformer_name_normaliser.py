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

        self._participants_norm_prefix = "ProteinMarker"
        self._other_entities_norm_prefix = "ProteinOther"

        self._random = Random(random_seed)

    def __call__(self, payload):
        other_entities = payload[self.other_entities_dict_key]
        raw_text = payload[self.text_key]

        # Get other replacements dict, offset, len, replacement
        entities_replacements = self._get_replacement(other_entities, raw_text, self._other_entities_norm_prefix)

        # Participant replacement dict
        participants_dict = self._get_participants_dict(payload)
        participants_replacements = self._get_replacement(participants_dict, raw_text, self._participants_norm_prefix)

        # Combine replacement for participants and other
        entities_replacements = entities_replacements + participants_replacements

        return self.name_replacer(text=raw_text, entities=entities_replacements)

    def _get_replacement(self, entities_dict_list, raw_text, norm_prefix):
        # Get replacement for non-participants
        unique_entity_names = set(
            [raw_text[e["charOffset"]:(e["charOffset"] + 1 + e["len"])] for e in entities_dict_list])

        entities_random_order = self._random.sample(list(unique_entity_names), k=len(unique_entity_names))
        entities_norm_replacement = {e: "{}{}".format(norm_prefix, i) for i, e in enumerate(entities_random_order)}

        entities_replacements = []

        for e in entities_dict_list:
            s_pos = e["charOffset"]
            e_pos = s_pos + 1 + e["len"]
            entity_name = raw_text[s_pos: e_pos]
            norm_replacement = entities_norm_replacement[entity_name]
            entities_replacements.append({
                "charOffset": e["charOffset"]
                , "len": e["len"]
                , "replacement": norm_replacement
            })
        return entities_replacements

    def _get_participants_dict(self, payload):
        result = []

        # Participant 1
        p1_start = payload[self.participant1_offset_key]
        p1_len = payload[self.participant1_len_key]

        result.append({
            "charOffset": p1_start,
            "len": p1_len
        }
        )

        # Participant 2
        p2_start = payload[self.participant2_offset_key]
        p2_len = payload[self.participant2_len_key]

        result.append({
            "charOffset": p2_start,
            "len": p2_len
        }
        )

        return result

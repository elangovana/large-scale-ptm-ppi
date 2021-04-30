from random import Random

from datasets.transformer_name_replacer import TransformerNameReplacer


class TransformerNameNormaliser:

    def __init__(self, text_key: str, participants_entities_dict_key: str, other_entities_dict_key: str,
                 name_replacer=None, random_seed=None):
        """
        Normalise entities to a predefined set
        :param participants_entities_dict_key: Participants entities dict key and this is a dict with at least these 2 keys
                {"charOffset": 62
                , "len": 4 }

        :param text_key: The name of the text field key

        :param other_entities_dict_key: Other entities key, and this is a dict with at least these 2 keys
                {"charOffset": 62
                , "len": 4 }
        :param name_replacer: An optional transformer to perform the replacement
        :param random_seed: Use random seed to fix the protein name replacements

        """
        self.participants_entities_dict_key = participants_entities_dict_key
        self.text_key = text_key

        # Other entities
        self.other_entities_dict_key = other_entities_dict_key

        self.name_replacer = name_replacer or TransformerNameReplacer()

        self._participants_norm_prefix = "PROTPART"
        self._other_entities_norm_prefix = "PRTIG"

        self._random = Random(random_seed)

    def __call__(self, payload):
        other_entities = payload[self.other_entities_dict_key]
        participants_dict = payload[self.participants_entities_dict_key]
        raw_text = payload[self.text_key]

        # Get other replacements dict, offset, len, replacement
        entities_replacements = self._get_replacement(other_entities, raw_text, self._other_entities_norm_prefix)

        # Participant replacement dict
        participants_replacements = self._get_replacement(participants_dict, raw_text, self._participants_norm_prefix)

        # Combine replacement for participants and other
        entities_replacements = entities_replacements + participants_replacements

        return self.name_replacer(text=raw_text, entities=entities_replacements)

    def _get_replacement(self, entities_dict_list, raw_text, norm_prefix):
        # Get replacement for non-participants
        unique_entity_names = sorted(set(
            [raw_text[e["charOffset"]:(e["charOffset"] + e["len"])] for e in entities_dict_list]))

        entities_random_order = self._random.sample(list(unique_entity_names), k=len(unique_entity_names))
        entities_norm_replacement = {e: "{}{}".format(norm_prefix, i) for i, e in enumerate(entities_random_order)}

        entities_replacements = []

        for e in entities_dict_list:
            s_pos = e["charOffset"]
            e_pos = s_pos + e["len"]
            entity_name = raw_text[s_pos: e_pos]
            norm_replacement = entities_norm_replacement[entity_name]
            entities_replacements.append({
                "charOffset": e["charOffset"]
                , "len": e["len"]
                , "replacement": norm_replacement
                , "original": entity_name
            })
        return entities_replacements

import logging
from typing import List, Dict


class TransformerNameReplacer:
    """
    Replaces named entity mentions in text with custom names
    """

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __call__(self, text: str, entities: List[Dict]) -> str:
        """
        Replaces named entity mentions in text with custom names
        :param text: The text to replace
        :param entities: A list of dictionary. The dictionary should contain the keys shown below
                         [{"charOffset": 62
                        , "len": 4
                        , "replacement": "Protein1"
                     }]
        :return: The replaced string
        """
        pos_sorted_entities = sorted(entities, key=lambda x: int(x["charOffset"]))



        adjustment_pos = 0
        previous_data_pos, previous_data_len = -1, -1
        last_replacement_end_position = 0

        replaced_text = text
        for entity_detail in pos_sorted_entities:

            s_orig_pos = int(entity_detail["charOffset"])
            s_pos = s_orig_pos + adjustment_pos
            entity_len = entity_detail["len"]
            e_pos = s_pos + int(entity_len)

            if s_orig_pos  <= (previous_data_pos+previous_data_len):
                self._logger.warning(
                    "Skip: The position {} overlaps with previous data pos {} including len {} \n{} \n{}".format(
                        s_orig_pos,
                        previous_data_pos,
                        previous_data_len,
                        text,
                        pos_sorted_entities
                    ))
                continue
            else:
                assert s_pos >= last_replacement_end_position, "Something has gone wrong..Start position {} is <= last_replacement_end_position {}, \n{}, \n{} \n{}" \
                    .format(s_pos, last_replacement_end_position, text,
                            pos_sorted_entities, s_orig_pos)

            replaced_substring = replaced_text[:s_pos] + entity_detail["replacement"]
            replaced_text = replaced_substring + replaced_text[e_pos:]
            previous_data_pos, previous_data_len = s_orig_pos, entity_len

            adjustment_pos += len(entity_detail["replacement"]) - (e_pos - s_pos)

            last_replacement_end_position = len(replaced_substring)

        return replaced_text

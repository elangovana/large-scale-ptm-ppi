import logging
from typing import List, Dict

from datasets.base_transformer import BaseTransformer


class TransformerNameReplacer(BaseTransformer):
    """
    Replaces named entity mentions in text with custom names
    """
    @property
    def _logger(self):
        return  logging.getLogger(__name__)

    def transform(self, *, text:str, entities:List[Dict]) -> str:
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

        last_replaced_position = 0
        adjustment_pos = 0
        replaced_text = text
        for entity_detail in pos_sorted_entities:
            s_pos = int(entity_detail["charOffset"]) + adjustment_pos
            e_pos = s_pos + int(entity_detail["len"])

            if s_pos < last_replaced_position:
                self._logger.warning("The position {} overlaps with previously replaced text by {} positions".format(s_pos, last_replaced_position))

            replaced_text = replaced_text[:s_pos] + entity_detail["replacement"] +  replaced_text[e_pos:]

            last_replaced_position = e_pos
            adjustment_pos += len(entity_detail["replacement"]) - (e_pos - s_pos )

        return replaced_text



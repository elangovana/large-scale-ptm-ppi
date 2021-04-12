import logging
from typing import List

from datasets.aimed_dataset import AimedDataset
from datasets.aimed_label_mapper import AimedLabelMapper
from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.transformer_chain import TransformerChain
from datasets.transformer_name_normaliser import TransformerNameNormaliser
from scorers.result_scorer_auc_binary_factory import ResultScorerAucBinaryFactory
from scorers.result_scorer_f1_binary_factory import ResultScorerF1BinaryFactory


class AimedDatasetFactory(BaseDatasetFactory):

    def get_scorers(self):
        scores = [
            ResultScorerAucBinaryFactory().get()
            , ResultScorerF1BinaryFactory().get()
        ]
        return scores

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_label_mapper(self, data=None, preprocessor=None, **kwargs):
        return AimedLabelMapper()

    def get_dataset(self, data, preprocessors=None, **kwargs):
        random_seed = self._get_value(kwargs, "protein_name_replacer_random_seed", None)
        random_seed = int(random_seed) if random_seed else random_seed
        transformer_list = [
            TransformerNameNormaliser(text_key="text"
                                      , participant1_offset_key="participant1Offset"
                                      , participant1_len_key="participant1Len"
                                      , participant2_offset_key="participant2Offset"
                                      , participant2_len_key="participant2Len"
                                      , other_entities_dict_key="otherEntities"
                                      , random_seed=random_seed
                                      )
        ]

        preprocessors = preprocessors or []
        if not isinstance(preprocessors, List):
            preprocessors = [preprocessors]

        # Add additional preprocessors
        transformer_list = transformer_list + preprocessors

        transformer_chain = TransformerChain(transformer_list)
        return AimedDataset(data, transformer=transformer_chain, label_transformer=self.get_label_mapper())

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

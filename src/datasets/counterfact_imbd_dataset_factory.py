import logging
from typing import List

from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.counterface_imdb_label_mapper import CounterfactImdbLabelMapper
from datasets.counterfact_imdb_dataset import CounterfactImdbDataset
from datasets.transformer_chain import TransformerChain
from scorers.result_scorer_acuracy_factory import ResultScorerAccuracyFactory


class CounterfactImdbDatasetFactory(BaseDatasetFactory):

    def get_scorers(self):
        scores = [
            ResultScorerAccuracyFactory().get()
        ]
        return scores

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_label_mapper(self, data=None, preprocessor=None, **kwargs):
        return CounterfactImdbLabelMapper()

    def get_dataset(self, data, preprocessors=None, **kwargs):
        preprocessors = preprocessors or []
        if not isinstance(preprocessors, List):
            preprocessors = [preprocessors]

        transformer_list = preprocessors

        transformer_chain = TransformerChain(transformer_list)
        return CounterfactImdbDataset(data, transformer=transformer_chain,
                                      label_transformer=self.get_label_mapper())

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

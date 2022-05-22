import logging
from typing import List

from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.chemprot_dataset import ChemprotDataset
from datasets.chemprot_label_mapper import ChemprotLabelMapper
from datasets.transformer_chain import TransformerChain
from scorers.result_scorer_auc_macro_factory import ResultScorerAucMacroFactory
from scorers.result_scorer_f1_macro_factory import ResultScorerF1MacroFactory


class ChemprotDatasetFactory(BaseDatasetFactory):

    def get_scorers(self):
        scores = [ResultScorerF1MacroFactory().get(),
                  ResultScorerAucMacroFactory().get()

                  ]
        return scores

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_label_mapper(self, data=None, preprocessor=None, **kwargs):
        return ChemprotLabelMapper()

    def get_dataset(self, data, preprocessors=None, **kwargs):
        preprocessors = preprocessors or []
        if not isinstance(preprocessors, List):
            preprocessors = [preprocessors]

        transformer_list = preprocessors

        transformer_chain = TransformerChain(transformer_list)
        return ChemprotDataset(data, transformer=transformer_chain, label_transformer=self.get_label_mapper())

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

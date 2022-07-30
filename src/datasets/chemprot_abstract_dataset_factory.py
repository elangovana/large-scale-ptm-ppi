import logging
from typing import List

from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.chemprot_abstract_dataset import ChemprotAbstractDataset
from datasets.chemprot_label_mapper import ChemprotLabelMapper
from datasets.transformer_chain import TransformerChain
from datasets.transformer_name_normaliser import TransformerNameNormaliser
from datasets.transformer_ppi_participant_augmentor import TransformerPPIParticipantAugmentor
from scorers.result_scorer_auc_macro_factory import ResultScorerAucMacroFactory
from scorers.result_scorer_f1_macro_factory import ResultScorerF1MacroFactory


class ChemprotAbstractDatasetFactory(BaseDatasetFactory):

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
        random_seed = self._get_value(kwargs, "protein_name_replacer_random_seed", None)
        random_seed = int(random_seed) if random_seed else random_seed
        transformer_list = [
            TransformerPPIParticipantAugmentor(participant1_key="participant1_id"
                                               , participant2_key="participant2_id"
                                               , annotations_dict_key="annotations"
                                               , result_key_participant="participantEntities"
                                               , result_key_other="otherEntities"
                                               ),
            TransformerNameNormaliser(text_key="abstract",
                                      participants_entities_dict_key="participantEntities",
                                      other_entities_dict_key="otherEntities",
                                      random_seed=random_seed)
        ]

        preprocessors = preprocessors or []
        if not isinstance(preprocessors, List):
            preprocessors = [preprocessors]

        # Add additional preprocessors
        transformer_list = transformer_list + preprocessors

        transformer_chain = TransformerChain(transformer_list)
        return ChemprotAbstractDataset(data, transformer=transformer_chain, label_transformer=self.get_label_mapper())

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

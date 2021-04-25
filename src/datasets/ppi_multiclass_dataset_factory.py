import logging
from typing import List

from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.ppi_multiclass_dataset import PpiMulticlassDataset
from datasets.ppi_multiclass_label_mapper import PpiMulticlassLabelMapper
from datasets.transformer_chain import TransformerChain
from datasets.transformer_name_normaliser import TransformerNameNormaliser
from datasets.transformer_ppi_participant_augmentor import TransformerPPIParticipantAugmentor
from scorers.result_scorer_auc_macro_factory import ResultScorerAucMacroFactory
from scorers.result_scorer_f1_macro_factory import ResultScorerF1MacroFactory


class PpiMulticlassDatasetFactory(BaseDatasetFactory):

    def get_scorers(self):
        scores = [ResultScorerAucMacroFactory().get()
            , ResultScorerF1MacroFactory().get()
                  ]
        return scores

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_label_mapper(self, data=None, preprocessor=None, **kwargs):
        return PpiMulticlassLabelMapper()

    def get_dataset(self, data, preprocessors=None, **kwargs):
        random_seed = self._get_value(kwargs, "protein_name_replacer_random_seed", None)
        random_seed = int(random_seed) if random_seed else random_seed
        transformer_list = [
            TransformerPPIParticipantAugmentor(participant1_key="participant1Id"
                                               , participant2_key="participant2Id"
                                               , annotations_dict_key="normalised_abstract_annotations"
                                               , result_key_participant="participantEntities"
                                               , result_key_other="otherEntities"
                                               ),
            TransformerNameNormaliser(text_key="normalised_abstract",
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
        return PpiMulticlassDataset(data, transformer=transformer_chain, label_transformer=self.get_label_mapper())

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

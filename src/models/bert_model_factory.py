import json
import logging

import transformers
from transformers import BertTokenizer

from datasets.transformer_bert_tokeniser import TransformerBertTokeniser
from models.base_model_factory import BaseModelFactory
from models.bert_checkpoint_manager import BertCheckpointManager
from models.bert_model import BertModel


class BertModelFactory(BaseModelFactory):

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_model(self, num_classes, checkpoint_dir=None, **kwargs):
        self._logger.info("Retrieving model")

        # If checkpoint file is available, load from checkpoint
        state_dict = self.get_checkpoint_manager().read(checkpoint_dir)

        fine_tune = bool(self._get_value(kwargs, "model_fine_tune", "0"))
        model_config = json.loads(self._get_value(kwargs, "model_config", "{}"))
        model_dir = self._get_value(kwargs, "pretrained_model", "bert-base-cased")

        if model_config:
            model_config = transformers.BertConfig(**model_config)

        network = BertModel(model_dir, num_classes, fine_tune=fine_tune, bert_config=model_config)

        if state_dict is not None:
            # Only load from BERT pretrained when no checkpoint is available
            self._logger.info("checkpoint models found")
            network.load_state_dict(state_dict)

        self._logger.info("Retrieving model complete")
        return network

    def get_checkpoint_manager(self, **kwargs):
        return BertCheckpointManager()

    def get_tokenisor(self, **kwargs):
        self._logger.info("Retrieving Tokeniser")

        max_seq_len = int(self._get_value(kwargs, "tokenisor_max_seq_len", "512"))
        do_lower_case = bool(self._get_value(kwargs, "tokenisor_lower_case", "0"))
        data_dir = self._get_value(kwargs, "tokenisor_data_dir", "bert-base-cased")

        tokenisor = BertTokenizer.from_pretrained(data_dir, do_lower_case=do_lower_case)

        preprocessor = TransformerBertTokeniser(max_feature_len=max_seq_len, tokeniser=tokenisor)

        return preprocessor

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

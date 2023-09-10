import json
import logging

import transformers
from transformers import RobertaTokenizer

from datasets.transformer_roberta_tokeniser import TransformerRobertaTokeniser
from models.base_model_factory import BaseModelFactory
from models.roberta_checkpoint_manager import RobertaCheckpointManager
from models.roberta_model import RobertaModel


class RobertaModelFactory(BaseModelFactory):

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_model(self, num_classes, checkpoint_dir=None, **kwargs):
        self._logger.info("Retrieving model")

        # If checkpoint file is available, load from checkpoint
        network = self.get_checkpoint_manager().read(checkpoint_dir)

        fine_tune = bool(int(self._get_value(kwargs, "model_fine_tune", "0")))

        if network is None:
            model_config = self._get_model_config(kwargs)
            model_dir = self._get_value(kwargs, "pretrained_model", None)
            # Only load from BERT pretrained when no checkpoint is available
            self._logger.info("No Checkpoint models found, using dir or config".format(model_dir, model_config))
            network = RobertaModel(model_dir, num_classes, fine_tune=fine_tune, roberta_config=model_config)

        self._logger.info("Retrieving model complete")
        return network

    def _get_model_config(self, kwargs):
        model_config = None
        config_file = self._get_value(kwargs, "model_config", None)

        # If model config is provided load it
        if config_file:
            with open(config_file, "r") as f:
                config = json.load(f)
            self._logger.info("Loading bert from config")
            model_config = transformers.RobertaConfig(**config)

        return model_config

    def get_checkpoint_manager(self, **kwargs):
        return RobertaCheckpointManager()

    def get_tokenisor(self, **kwargs):
        self._logger.info("Retrieving Tokeniser")

        max_seq_len = int(self._get_value(kwargs, "tokenisor_max_seq_len", "512"))
        data_dir = self._get_value(kwargs, "tokenisor_data_dir",
                                   self._get_value(kwargs, "pretrained_model", "roberta-base"))

        tokenisor = RobertaTokenizer.from_pretrained(data_dir)

        preprocessor = TransformerRobertaTokeniser(max_feature_len=max_seq_len, tokeniser=tokenisor)

        return preprocessor

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

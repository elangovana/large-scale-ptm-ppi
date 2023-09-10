import logging

from torch import nn
from transformers import RobertaForSequenceClassification


class RobertaModel(nn.Module):

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __init__(self, model_name_or_dir, num_classes, fine_tune=True, roberta_config=None):
        super().__init__()
        assert model_name_or_dir is not None or roberta_config is not None, "Either a name or directory containing a pretrained model or a custom roberta config must be provided"

        if model_name_or_dir:
            self._logger.info("Loading RoBERTa from {}".format(model_name_or_dir))
            self.model = RobertaForSequenceClassification.from_pretrained(model_name_or_dir, num_labels=num_classes)
        else:
            self._logger.info("Initialing RoBERTa from config")
            self.model = RobertaForSequenceClassification(config=roberta_config)

        # Fine tune, freeze all other weights except classifier
        if fine_tune:
            self._freeze_base_weights()

    def _freeze_base_weights(self):
        self._logger.info("Freezing weights for base model")
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, *input):
        return self.model(*input)

    def save_pretrained(self, model_directory):
        self.model.save_pretrained(model_directory)

    @staticmethod
    def from_pretrained(model_name_or_dir, *args):
        return RobertaForSequenceClassification.from_pretrained(model_name_or_dir, *args)

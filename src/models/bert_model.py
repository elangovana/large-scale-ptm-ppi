import logging

from torch import nn
from transformers import BertForSequenceClassification


class BertModel(nn.Module):

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __init__(self, model_name_or_dir, num_classes, fine_tune=True, bert_config=None):
        super().__init__()
        assert model_name_or_dir is not None or bert_config is not None, "Either a name or directory containing a pretrained model or a custom bert config must be provided"

        if model_name_or_dir:
            self._logger.info("Loading BERT from {}".format(model_name_or_dir))
            self.model = BertForSequenceClassification.from_pretrained(model_name_or_dir, num_labels=num_classes)
        else:
            self._logger.info("Initialing BERT from config")
            self.model = BertForSequenceClassification(config=bert_config)

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
        return BertForSequenceClassification.from_pretrained(model_name_or_dir, *args)

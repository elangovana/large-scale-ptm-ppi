import logging

import torch


class TransformerRobertaTokeniser:
    """
    Text to an array of indices using the BERT tokeniser
    """

    def __init__(self, max_feature_len, tokeniser):
        self.max_feature_len = max_feature_len
        self.tokeniser = tokeniser
        self.item = None

    @staticmethod
    def pad_token():
        return "<pad>"

    def __call__(self, item):
        self.item = item
        self.tokenise() \
            .sequence_pad() \
            .token_to_index() \
            .to_tensor()

        return self.item

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def tokenise(self):
        """
        Converts text to tokens, e.g. "The dog" would return ["The", "dog"]
        """
        self._logger.debug("Text : {}".format(self.item))
        tokens = self.tokeniser.tokenize(self.item)
        self._logger.debug("Tokens : {}".format(tokens))

        self.item = tokens
        return self

    def token_to_index(self):
        """
        Converts a string of token to corresponding indices. e.g. ["The", "dog"] would return [2,3]
        :return: self
        """
        result = self.tokeniser.convert_tokens_to_ids(self.item)
        self.item = result

        return self

    def sequence_pad(self):
        """
        Converts the tokens to fixed size and formats it according to bert
        :return: self
        """
        tokens = self.item[:self.max_feature_len - 2]
        pad_tokens = [self.pad_token()] * (self.max_feature_len - 2 - len(tokens))
        result = ['<s>'] + tokens + pad_tokens + ['</s>']

        self.item = result
        return self

    def to_tensor(self):
        """
        Converts list of int to tensor
        :return: self
        """

        self.item = torch.tensor(self.item)
        return self

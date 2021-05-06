import logging

from torch import nn
from torch.optim import Adam

from bert_train import BertTrain
from locator import Locator


class TrainBuilder:

    def __init__(self, model_factory_name, model_dir, num_classes, scorers,
                 checkpoint_dir=None,
                 epochs=10,
                 early_stopping_patience=10, checkpoint_frequency=1, grad_accumulation_steps=8, learning_rate=0.00001,
                 use_loss_eval=False, addition_args_dict=None):
        self.scorers = scorers
        self.num_classes = num_classes
        self.use_loss_eval = use_loss_eval
        self._addition_args_dict = addition_args_dict
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir

        self.model_factory = Locator().get(model_factory_name)

        self._network = None

        self._trainer = None
        self._lossfunc = None
        self._optimiser = None

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_network(self):
        # If network already loaded simply return
        if self._network is not None: return self._network

        self._network = self.model_factory.get_model(num_classes=self.num_classes,
                                                     **self._addition_args_dict)

        self._logger.info("Retrieving model complete")

        return self._network

    def get_loss_function(self):
        if self._lossfunc is None:
            self._lossfunc = nn.CrossEntropyLoss()
        return self._lossfunc

    def get_optimiser(self):
        if self._optimiser is None:
            weight_decay = float(self._get_value(self._addition_args_dict, "weight_decay", "0.0"))
            self._optimiser = Adam(params=self.get_network().parameters(), lr=self.learning_rate,
                                   weight_decay=weight_decay)
        return self._optimiser

    def get_trainer(self):
        if self._trainer is None:
            self._trainer = BertTrain(model_dir=self.model_dir, scorers=self.scorers,
                                      epochs=self.epochs, early_stopping_patience=self.early_stopping_patience,
                                      checkpoint_frequency=self.checkpoint_frequency,
                                      checkpoint_dir=self.checkpoint_dir,
                                      accumulation_steps=self.grad_accumulation_steps,
                                      use_loss_eval=self.use_loss_eval,
                                      checkpoint_manager=self.model_factory.get_checkpoint_manager())

        return self._trainer

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

import logging
import os

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from bert_train import BertTrain
from locator import Locator


class Builder:

    def __init__(self, train_data, val_data, dataset_factory_name, model_factory_name, model_dir, num_workers=None,
                 checkpoint_dir=None,
                 epochs=10,
                 early_stopping_patience=10, checkpoint_frequency=1, grad_accumulation_steps=8, batch_size=8,
                 max_seq_len=512, learning_rate=0.00001, fine_tune=True, addition_args_dict=None):
        self._addition_args_dict = addition_args_dict
        self.model_dir = model_dir
        self.fine_tune = fine_tune
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.dataset_factory = Locator().get(dataset_factory_name)
        self.model_factory = Locator().get(model_factory_name)

        self._max_seq_len = max_seq_len
        self.num_workers = num_workers or os.cpu_count() - 1
        if self.num_workers <= 0:
            self.num_workers = 0

        self._network = None
        self._tokenisor = None
        self._train_dataloader = None
        self._train_dataset = None
        self._val_dataset = None
        self._val_dataloader = None
        self._trainer = None
        self._lossfunc = None
        self._optimiser = None
        self._label_mapper = None

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_tokenisor(self):
        self._logger.info("Retrieving Tokeniser")

        if self._tokenisor is None:
            self._tokenisor = self.model_factory.get_tokenisor(**self._addition_args_dict)

        return self._tokenisor

    def get_network(self):
        # If network already loaded simply return
        if self._network is not None: return self._network

        self._network = self.model_factory.get_model(num_classes=self.get_label_mapper().num_classes,
                                                     **self._addition_args_dict)

        self._logger.info("Retrieving model complete")

        return self._network

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.dataset_factory.get_dataset(self.train_data,
                                                                   preprocessors=self.get_tokenisor())

        return self._train_dataset

    def get_val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = self.dataset_factory.get_dataset(self.val_data, preprocessors=self.get_tokenisor())

        return self._val_dataset

    def get_label_mapper(self):
        if self._label_mapper is None:
            self._label_mapper = self.dataset_factory.get_label_mapper()

        return self._label_mapper

    def get_pos_label_index(self):
        return self.get_label_mapper().positive_label_index

    def get_train_val_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(dataset=self.get_train_dataset(), num_workers=self.num_workers,
                                                batch_size=self.batch_size, shuffle=True)

        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(dataset=self.get_val_dataset(), num_workers=self.num_workers,
                                              batch_size=self.batch_size, shuffle=False)

        return self._train_dataloader, self._val_dataloader

    def get_loss_function(self):
        if self._lossfunc is None:
            self._lossfunc = nn.CrossEntropyLoss()
        return self._lossfunc

    def get_optimiser(self):
        if self._optimiser is None:
            self._optimiser = Adam(params=self.get_network().parameters(), lr=self.learning_rate)
        return self._optimiser

    def get_trainer(self):
        if self._trainer is None:
            self._trainer = BertTrain(model_dir=self.model_dir, scorer=self.dataset_factory.get_scorers(),
                                      epochs=self.epochs, early_stopping_patience=self.early_stopping_patience,
                                      checkpoint_frequency=self.checkpoint_frequency,
                                      checkpoint_dir=self.checkpoint_dir,
                                      accumulation_steps=self.grad_accumulation_steps,
                                      checkpoint_manager=self.model_factory.get_checkpoint_manager())

        return self._trainer

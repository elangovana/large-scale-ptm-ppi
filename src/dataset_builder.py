import logging
import os

from torch.utils.data import DataLoader

from locator import Locator


class DatasetBuilder:

    def __init__(self, val_data, dataset_factory_name, tokenisor_factory_name, train_data=None, num_workers=None,
                 batch_size=8, addition_args_dict=None):
        self._addition_args_dict = addition_args_dict

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self._dataset_factory = Locator().get(dataset_factory_name)
        self._tokenisor_factory = Locator().get(tokenisor_factory_name)

        self.num_workers = num_workers or os.cpu_count() - 1
        if self.num_workers <= 0:
            self.num_workers = 0

        self._tokenisor = None
        self._train_dataloader = None
        self._train_dataset = None
        self._val_dataset = None
        self._val_dataloader = None
        self._scorers = None
        self._label_mapper = None

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_tokenisor(self):
        self._logger.info("Retrieving Tokeniser")

        if self._tokenisor is None:
            self._tokenisor = self._tokenisor_factory.get_tokenisor(**self._addition_args_dict)

        return self._tokenisor

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self._dataset_factory.get_dataset(self.train_data,
                                                                    preprocessors=self.get_tokenisor(),
                                                                    **self._addition_args_dict)

        return self._train_dataset

    def get_val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = self._dataset_factory.get_dataset(self.val_data, preprocessors=self.get_tokenisor(),
                                                                  **self._addition_args_dict)

        return self._val_dataset

    def get_label_mapper(self):
        if self._label_mapper is None:
            self._label_mapper = self._dataset_factory.get_label_mapper()

        return self._label_mapper

    def num_classes(self):
        return self.get_label_mapper().num_classes

    def positive_label_index(self):
        return self._label_mapper.positive_label_index

    def get_scorers(self):
        if self._scorers is None:
            self._scorers = self._dataset_factory.get_scorers()

        return self._scorers

    def get_train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(dataset=self.get_train_dataset(), num_workers=self.num_workers,
                                                batch_size=self.batch_size, shuffle=True)

        return self._train_dataloader

    def get_val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(dataset=self.get_val_dataset(), num_workers=self.num_workers,
                                              batch_size=self.batch_size, shuffle=False)
        return self._val_dataloader

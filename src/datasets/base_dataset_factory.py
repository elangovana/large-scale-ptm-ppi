class BaseDatasetFactory:
    def get_dataset(self, data=None, preprocessors=None, **kwargs):
        raise NotImplementedError

    def get_label_mapper(self, data=None, preprocessors=None, **kwargs):
        raise NotImplementedError
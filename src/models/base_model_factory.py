from models.base_tokenisor_factory import BaseTokenisorFactory


class BaseModelFactory(BaseTokenisorFactory):

    def get_checkpoint_manager(self, **kwargs):
        raise NotImplementedError

    def get_model(self, num_classes, checkpoint_dir=None, **kwargs):
        raise NotImplementedError

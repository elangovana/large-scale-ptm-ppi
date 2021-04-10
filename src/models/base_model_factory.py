class BaseModelFactory:

    def get_checkpoint_manager(self, **kwargs):
        raise NotImplementedError

    def get_model(self, num_classes, checkpoint_dir=None, **kwargs):
        raise NotImplementedError

    def get_tokenisor(self, **kwargs):
        raise NotImplementedError

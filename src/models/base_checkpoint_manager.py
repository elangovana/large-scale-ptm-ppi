class BaseCheckpointManager:

    def read(self, checkpoint_dir, **kwargs):
        raise NotImplementedError

    def write(self, model, checkpoint_dir, **kwargs):
        raise NotImplementedError

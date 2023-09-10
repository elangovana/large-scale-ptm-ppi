import logging
import os

from models.base_checkpoint_manager import BaseCheckpointManager
from models.roberta_model import RobertaModel


class RobertaCheckpointManager(BaseCheckpointManager):

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def read(self, checkpoint_dir, **kwargs):
        model = None
        self._logger.info("Attempting to load checkpoint from {}".format(checkpoint_dir))

        # Make sure there are contents in the checkpoint dir
        has_checkpoint = bool(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0

        if has_checkpoint:
            model = RobertaModel.from_pretrained(checkpoint_dir)

        return model

    def write(self, model, checkpoint_dir, **kwargs):
        self._logger.info("Checkpoint model to {}".format(checkpoint_dir))

        model.save_pretrained(checkpoint_dir)

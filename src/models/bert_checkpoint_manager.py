import glob
import logging
import os

import torch


class BertCheckpointManager:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def read(self, checkpoint_dir, **kwargs):
        loaded_weights = None
        if checkpoint_dir is not None:
            model_files = list(glob.glob("{}/*.pt".format(checkpoint_dir)))
            if len(model_files) > 0:
                model_file = model_files[0]
                self._logger.info(
                    "Loading checkpoint {} , found {} checkpoint files".format(model_file, len(model_files)))
                checkpoint = torch.load(model_file)
                loaded_weights = checkpoint['model_state_dict']

        return loaded_weights

    def write(self, model, checkpoint_dir, **kwargs):
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

        self._logger.info("Checkpoint model to {}".format(checkpoint_path))

        # If nn.dataparallel, get the underlying module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        torch.save({
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)

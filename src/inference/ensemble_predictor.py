import logging
from multiprocessing.dummy import Pool

import torch

from inference.predictor import Predictor


class EnsemblePredictor:

    def __init__(self, model_wrapper=None):
        self.model_wrapper = model_wrapper or Predictor()

    def predict(self, model_networks, dataloader, device=None):
        if not self._is_iterable(model_networks):
            model_networks = [model_networks]

        if device is None:
            if torch.cuda.device_count() > 0:
                devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]

                # This is a hack workaround to initialise the device before multi threading
                # See issue https://github.com/pytorch/pytorch/issues/16559
                for i in range(len(devices)):
                    with torch.cuda.device(i):
                        torch.tensor([1.]).cuda()
            else:
                devices = ["cpu"]
        else:
            devices = [device]

        # Use all available GPUS using multithreading
        self._logger.info("Using devices {}".format(devices))
        model_device_map = ((m.to(devices[i % len(devices)]), dataloader, devices[i % len(devices)]) for i, m in
                            enumerate(model_networks))
        with Pool(len(devices)) as p:
            agg_pred_scores = p.starmap(self.model_wrapper.predict, model_device_map)

        # Compute average
        self._logger.info("Computing average ")
        ensemble_tensor = torch.cat([torch.unsqueeze(s, dim=0) for _, s in agg_pred_scores], dim=0)
        average_scores_ensemble = torch.mean(ensemble_tensor, dim=0)
        std_scores_ensemble = torch.std(ensemble_tensor, dim=0)

        # Predicted ensemble , arg max
        self._logger.info("Computing ensemble prediction ")
        predicted_ensemble = torch.max(average_scores_ensemble, dim=-1)[1].view(-1)

        conf_indices = [[i for i, _ in enumerate(predicted_ensemble)], predicted_ensemble.cpu().tolist()]
        predicted_ensemble_conf_scores = ensemble_tensor.permute(1, 2, 0)[conf_indices]

        self._logger.info("Completed ensemble prediction ")
        return predicted_ensemble, average_scores_ensemble, std_scores_ensemble, predicted_ensemble_conf_scores

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @staticmethod
    def _is_iterable(o):
        try:
            iter(o)
        except TypeError:
            return False
        else:
            return True

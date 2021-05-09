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
        model_device_map = [(m, dataloader, devices[i % len(devices)]) for i, m in enumerate(model_networks)]
        with Pool(len(devices)) as p:
            agg_pred_scores = p.starmap(self.model_wrapper.predict, model_device_map)

        # Compute average
        self._logger.info("Computing average ")
        ensemble_size = len(agg_pred_scores)
        _, scores_ensemble = agg_pred_scores[0]
        scores_ensemble.to(device="cpu")
        for _, s in agg_pred_scores[1:]:
            scores_ensemble = scores_ensemble + s.to(device=scores_ensemble.device)
        scores_ensemble = scores_ensemble / ensemble_size

        # Predicted ensemble , arg max
        self._logger.info("Computing ensemble prediction ")
        predicted_ensemble = torch.max(scores_ensemble, dim=-1)[1].view(-1)

        self._logger.info("Comppleted ensemble prediction ")
        return predicted_ensemble, scores_ensemble

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

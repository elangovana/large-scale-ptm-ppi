import logging

import torch


class Predictor:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def predict(self, model_network, dataloader, device=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.logger.info("Using device {}".format(device))
        model_network.to(device)
        # switch model to evaluation mode
        model_network.eval()
        self.logger.info("Running inference {}".format(device))
        scores = []

        with torch.no_grad():
            soft_max_func = torch.nn.Softmax(dim=-1)
            for _, (batch_x, batch_y) in enumerate(dataloader):

                # TODO: CLean this up
                if isinstance(batch_x, list):
                    val_batch_idx = [t.to(device=device) for t in batch_x]
                else:
                    val_batch_idx = batch_x.to(device=device)

                pred_batch_y = model_network(val_batch_idx)[0]
                # Soft max the predictions
                pred_batch_y = soft_max_func(pred_batch_y)

                # Copy to CPU to release gpu mem...
                scores.append(pred_batch_y.cpu())

        scores = torch.cat(scores)
        predicted = torch.max(scores, dim=-1)[1].view(-1)

        self.logger.info("Completed inference {}".format(device))

        return predicted, scores

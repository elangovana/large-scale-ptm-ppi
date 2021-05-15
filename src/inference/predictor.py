import logging

import torch


class Predictor:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def predict(self, model_network, dataloader, device=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.logger.info("Using device {}".format(device))

        scores = []

        with torch.no_grad():
            model_network.to(device)
            # switch model to evaluation mode
            model_network.eval()

            soft_max_func = torch.nn.Softmax(dim=-1)
            for i, (batch_x, batch_y) in enumerate(dataloader):
                self.logger.debug("running batch {}".format(i))
                # TODO: CLean this up
                if isinstance(batch_x, list):
                    val_batch_idx = [t.to(device=device) for t in batch_x]
                else:
                    val_batch_idx = batch_x.to(device=device)
                self.logger.debug("predict batch {}".format(i))

                pred_batch_y = model_network(val_batch_idx)[0]

                self.logger.debug("softmax batch {}".format(i))

                # Soft max the predictions
                pred_batch_y = soft_max_func(pred_batch_y)

                self.logger.debug("copy cpu {}".format(i))

                # Copy to CPU to release gpu mem...
                scores.append(pred_batch_y.cpu())
                self.logger.debug("Completed cpu {}".format(i))
            self.logger.debug("In grad {}".format(i))

        self.logger.debug("running concat {}".format(device))

        scores_tensor = torch.cat(scores)
        predicted = torch.max(scores_tensor, dim=-1)[1].view(-1)

        self.logger.info("Completed inference {}".format(device))

        return predicted, scores_tensor

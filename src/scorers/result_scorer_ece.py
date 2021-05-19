import numpy as np

from scorers.base_classification_scorer import BaseClassificationScorer


class ResultScorerEce(BaseClassificationScorer):
    """

    Calculate the score Expected Calibrated Error score
    """

    def __init__(self, num_bins=10):
        self.num_bins = num_bins

    def __call__(self, y_actual, y_pred, pos_label=None):
        bin_values = self.get_bin_values(y_actual, y_pred)

        expected_accuracy_bins = bin_values["expected_accuracy"]
        actual_accuracy_bins = bin_values["actual_accuracy"]
        num_items_in_bin = bin_values["num_items_in_bin"]
        average_confidence_bins = bin_values["average_confidence"]
        n = len(y_actual)

        ece_error = 0
        for i in range(len(expected_accuracy_bins)):
            conf_bin = average_confidence_bins[i]
            acc_bin = actual_accuracy_bins[i]
            bin_size = num_items_in_bin[i]
            ece_error = ece_error + abs(conf_bin - acc_bin) * bin_size / n
        return ece_error

    def get_bin_values(self, y_actual, y_pred):
        assert len(y_pred.shape) == 2, "Only Accept 2D "

        y_pred_conf = np.array(y_pred)
        y_actual = np.array(y_actual)

        assert 0 <= np.min(y_pred_conf) <= 1, "Confidence score must range between 0 and 1, but min value is {}".format(
            np.min(y_pred_conf))

        assert 0 <= np.max(y_pred_conf) <= 1, "Confidence score must range between 0 and 1, but max value is {}".format(
            np.max(y_pred_conf))

        bins = self._get_bins(0, 1)

        actual_accuracy_bins = []
        expected_accuracy_bins = []
        items_in_bin = []
        avg_conf_bins = []

        for s, e in zip(bins[:-1], bins[1:]):
            accuracy_bin, expected_accuracy, num_item, bin_avg_conf = self._bin_accuracy(y_actual, y_pred_conf, s, e)
            actual_accuracy_bins.append(accuracy_bin)
            expected_accuracy_bins.append(expected_accuracy)
            items_in_bin.append(num_item)
            avg_conf_bins.append(bin_avg_conf)

        return {"bins": bins,
                "expected_accuracy": expected_accuracy_bins,
                "actual_accuracy": actual_accuracy_bins,
                "average_confidence": avg_conf_bins,
                "num_items_in_bin": items_in_bin}

    def _get_bins(self, pred_confidence_min, pred_confidence_max):
        bin_size = (pred_confidence_max - pred_confidence_min) / self.num_bins
        bins = [bin_size * i for i in range(0, self.num_bins + 1)]

        return bins

    def _bin_accuracy(self, y_actual, y_pred, conf_min, conf_max):

        filter_in_range = filter(lambda ix: conf_min <= np.max(ix[1]) <= conf_max, enumerate(y_pred))
        index_in_range = list(map(lambda ix: ix[0], filter_in_range))

        bin_expected_accuracy = (conf_max + conf_min) / 2

        items_in_bin = len(index_in_range)

        item_in_y_actual = y_actual[index_in_range]

        item_in_y_pred = y_pred[index_in_range]

        predictions = np.argmax(item_in_y_pred, axis=-1)
        actual = item_in_y_actual

        # compute mean, if no items then set accuracy to 0
        bin_accuracy = np.mean(predictions == actual)

        bin_average_confidence = np.mean(np.max(item_in_y_pred, axis=-1))

        # if no items then set accuracy to 0
        if len(predictions) == 0:
            bin_accuracy = 0
            bin_average_confidence = 0

        result = bin_accuracy, bin_expected_accuracy, items_in_bin, bin_average_confidence

        return result

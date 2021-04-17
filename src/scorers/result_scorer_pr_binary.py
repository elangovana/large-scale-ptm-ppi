import numpy as np
from sklearn.metrics import average_precision_score

from scorers.base_classification_scorer import BaseClassificationScorer


class ResultScorerPrBinary(BaseClassificationScorer):
    """

    Calculate the score PR score
    """

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        y_pred = np.array(y_pred)
        y_actual = np.array(y_actual)

        # if 2 D array, get max label index
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:, pos_label]

        score = average_precision_score(y_actual, y_pred)

        return score

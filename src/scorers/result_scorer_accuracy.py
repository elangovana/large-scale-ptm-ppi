import numpy as np
from sklearn.metrics import accuracy_score

from scorers.base_classification_scorer import BaseClassificationScorer


class ResultScorerAccuracy(BaseClassificationScorer):
    """

    Calculate the score AUC score
    """

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        y_pred = np.array(y_pred)
        y_actual = np.array(y_actual)

        # if 2 D array, get max label index
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=-1)

        accuracy = accuracy_score(y_actual, y_pred)

        return accuracy

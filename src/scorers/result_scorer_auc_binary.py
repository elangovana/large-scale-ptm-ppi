from scorers.base_classification_scorer import BaseClassificationScorer
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class ResultScorerAucBinary(BaseClassificationScorer):
    """

    Calculate the score AUC score
    """

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        y_pred = np.array(y_pred)
        y_actual = np.array(y_actual)

        f1 = roc_auc_score(y_actual, y_pred, average='binary')

        return f1

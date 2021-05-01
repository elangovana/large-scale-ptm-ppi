import numpy as np

from scorers.base_classification_scorer import BaseClassificationScorer


class ResultScorerF1Macro(BaseClassificationScorer):
    """

    Calculate the score F1 Macro
    """
    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        from sklearn.metrics import f1_score

        y_pred = np.array(y_pred)
        # if 2 D array, get max label index
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=-1)

        f1 = f1_score(y_actual, y_pred, pos_label=pos_label, average='macro')

        return f1

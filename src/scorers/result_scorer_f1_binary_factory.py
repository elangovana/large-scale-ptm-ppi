from scorers.base_classification_scorer_factory import BaseClassificationScorerFactory
from scorers.result_scorer_f1_binary import ResultScorerF1Binary


class ResultScorerF1BinaryFactory(BaseClassificationScorerFactory):
    """
    Factory for F1 results_scorer
    """

    def get(self):
        return ResultScorerF1Binary()

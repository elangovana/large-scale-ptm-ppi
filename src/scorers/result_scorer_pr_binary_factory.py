from scorers.base_classification_scorer_factory import BaseClassificationScorerFactory
from scorers.result_scorer_pr_binary import ResultScorerPrBinary


class ResultScorerPrBinaryFactory(BaseClassificationScorerFactory):
    """
    Factory for Pr Binary
    """

    def get(self):
        return ResultScorerPrBinary()

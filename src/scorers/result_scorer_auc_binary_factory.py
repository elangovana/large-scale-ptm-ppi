from scorers.base_classification_scorer_factory import BaseClassificationScorerFactory
from scorers.result_scorer_auc_binary import ResultScorerAucBinary
from scorers.result_scorer_auc_macro import ResultScorerAucMacro
from scorers.result_scorer_f1_macro import ResultScorerF1Macro


class ResultScorerAucBinaryFactory(BaseClassificationScorerFactory):
    """
    Factory for AUB Binary
    """

    def get(self):
        return ResultScorerAucBinary()

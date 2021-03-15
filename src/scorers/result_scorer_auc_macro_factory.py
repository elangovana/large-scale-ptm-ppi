from scorers.base_classification_scorer_factory import BaseClassificationScorerFactory
from scorers.result_scorer_auc_macro import ResultScorerAucMacro
from scorers.result_scorer_f1_macro import ResultScorerF1Macro


class ResultScorerAucMacroFactory(BaseClassificationScorerFactory):
    """
    Factory for F1 results_scorer
    """

    def get(self):
        return ResultScorerAucMacro()

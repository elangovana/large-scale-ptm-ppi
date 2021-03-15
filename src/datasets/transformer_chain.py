from typing import List



class TransformerChain:
    """
    Chains a list of transform operations.
    """

    def __init__(self, transformers: List):
        self._transformers = transformers

    def __call__(self, data):
        transformed_result = data
        for transformer in self._transformers:
            transformed_result = transformer(transformed_result)

        return transformed_result

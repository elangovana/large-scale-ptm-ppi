class BaseTransformer:
    """
    Transforms a single record
    """
    def transform(self, **kwargs):
        raise NotImplementedError
from datasets.base_label_mapper import BaseMapperBase


class ChemprotAdverserialLabelMapper(BaseMapperBase):
    """
    Maps string labels to integers for Chemprot dataaset
    """

    def map(self, item) -> int:
        return item

    def reverse_map(self, item: int):
        return item

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def positive_label(self):
        return self.reverse_map(self.positive_label_index)

    @property
    def positive_label_index(self) -> int:
        return 1

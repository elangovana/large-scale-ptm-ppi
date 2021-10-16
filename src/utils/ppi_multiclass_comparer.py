from datasets.ppi_multiclass_dataset_factory import PpiMulticlassDatasetFactory
from utils.similarity_comparer import SimilarityComparer


class PpiMulticlassComparer:

    def __init__(self, comparer=None):
        self._comparer = comparer or SimilarityComparer()

    def compare(self, reference_file, new_file):
        dataset_factory = PpiMulticlassDatasetFactory()
        ref_data = dataset_factory.get_dataset(reference_file)
        new_data = dataset_factory.get_dataset(new_file)

        # Retain original labels without the index
        ref_data.label_transformer = None
        new_data.label_transformer = None

        ref_labels = list(set([y_label for _, y_label in ref_data]))

        result = {}
        for y_label in ref_labels:
            ref_data_list = [x for x, y in ref_data if y == y_label]
            new_data_list = [x for x, y in new_data if y == y_label]

            if len(new_data_list) == 0: continue
            result[y_label] = self._comparer(ref_data_list, new_data_list)

        return result

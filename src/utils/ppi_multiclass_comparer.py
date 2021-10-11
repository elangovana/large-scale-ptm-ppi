from datasets.ppi_multiclass_dataset_factory import PpiMulticlassDatasetFactory
from utils.similarity_comparer import SimilarityComparer


class PpiMulticlassComparer:

    def __init__(self, comparer=None):
        self._comparer = comparer or SimilarityComparer()

    def compare(self, reference_file, new_file):
        dataset_factory = PpiMulticlassDatasetFactory()
        ref_data = dataset_factory.get_dataset(reference_file)
        new_file = dataset_factory.get_dataset(new_file)

        ref_data_list = [x for x, _ in ref_data]
        new_data_list = [x for x, _ in new_file]

        return self._comparer(ref_data_list, new_data_list)

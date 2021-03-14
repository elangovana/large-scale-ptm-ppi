from torch.utils.data import Dataset
import pandas as pd


class AimedDataset(Dataset):
    """
    Loads an AIMed dataset formatted as Json.
    """

    def __init__(self, file_path_or_dataframe, transformer=None):
        self._file_path = file_path_or_dataframe
        self.transformer = transformer
        # Read json
        if isinstance(file_path_or_dataframe, str):
            data_df = pd.read_json(self._file_path)
        elif isinstance(file_path_or_dataframe, pd.DataFrame):
            data_df = file_path_or_dataframe
        else:
            raise ValueError(
                "The type of argument file_path_or_dataframe  must be a str or pandas dataframe, but is {}".format(
                    type(file_path_or_dataframe)))

        # Filter features
        self._data_df = data_df[
            ["text", "participant1Offset", "participant1Len", "participant2Offset", "participant2Len", "otherEntities"]]

        # Set up labels
        if "interacts" in data_df.columns:
            self._labels = data_df["interacts"].to_list()
        else:
            self._labels = [False for _ in range(data_df.shape[0])]

    def __len__(self):
        return self._data_df.shape[0]

    def __getitem__(self, index):
        row_values = self._data_df.iloc[index, :].to_dict()

        # transform
        if self.transformer is not None:
            row_values = self.transformer(row_values)

        x = row_values
        y = self._labels[index]
        return x, y

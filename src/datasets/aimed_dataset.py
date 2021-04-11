import os

import pandas as pd
from torch.utils.data import Dataset


class AimedDataset(Dataset):
    """
    Loads an AIMed dataset formatted as Json.
    """

    def __init__(self, path_or_dataframe, transformer=None, label_transformer=None):
        self.transformer = transformer
        self.label_transformer = label_transformer

        # Read json from path
        if isinstance(path_or_dataframe, str):
            file_path = path_or_dataframe
            # If directory make sure there is just one file in it
            if os.path.isdir(path_or_dataframe):
                files_in_dir = os.listdir(path_or_dataframe)
                assert len(
                    files_in_dir) == 1, "Expecting exactly one file in the path_or_dataframe, but found {}".format(
                    files_in_dir)
                file_path = os.path.join(path_or_dataframe, files_in_dir[0])
            data_df = pd.read_json(file_path)
        # Else read from data frame
        elif isinstance(path_or_dataframe, pd.DataFrame):
            data_df = path_or_dataframe
        else:
            raise ValueError(
                "The type of argument file_path_or_dataframe  must be a str or pandas dataframe, but is {}".format(
                    type(path_or_dataframe)))

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
        x = self._data_df.iloc[index, :].to_dict()
        y = self._labels[index]

        # transform
        if self.transformer is not None:
            x = self.transformer(x)

        if  self.label_transformer is not None:
            y = self.label_transformer.map(y)

        return x, y

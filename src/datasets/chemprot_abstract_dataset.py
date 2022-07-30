import os

import pandas as pd
from torch.utils.data import Dataset


class ChemprotAbstractDataset(Dataset):
    """
    Represents Chemprot dataset without abstract preprocessing
    """

    def __init__(self, path_or_dataframe, interaction_type=None, transformer=None, label_transformer=None):
        self.label_transformer = label_transformer
        self.transformer = transformer
        self._file_path = path_or_dataframe

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

        # Filter interaction types if required
        if interaction_type is not None:
            data_df = data_df.query('interactionType == "{}"'.format(interaction_type))

        # Filter features
        self._data_df = data_df[
            ["abstract", "participant1_id", "participant2_id", "annotations"]]

        # Set up labels
        if "label" in data_df.columns:
            self._labels = data_df["label"].tolist()
        else:
            self._labels = ["NEGATIVE" for _ in range(data_df.shape[0])]

    def __len__(self):
        return self._data_df.shape[0]

    def __getitem__(self, index):
        x = self._data_df.iloc[index, :].to_dict()
        y = self._labels[index]

        if self.transformer:
            x = self.transformer(x)

        if self.label_transformer:
            y = self.label_transformer.map(y)

        return x, y

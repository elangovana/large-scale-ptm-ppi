import logging
import os
from _csv import QUOTE_NONE

import pandas as pd
from torch.utils.data import Dataset


class ChemprotSelfsupervisedDataset(Dataset):
    """
    Represents  Chemprot Self supervised
    """

    def __init__(self, path_or_dataframe, transformer=None, label_transformer=None):
        self.label_transformer = label_transformer
        self.transformer = transformer
        self._file_path = path_or_dataframe

        default_df_reader = pd.read_json
        self._df_reader_map = {
            ".tsv": lambda x: pd.read_csv(x, delimiter='\t', quotechar=None, quoting=QUOTE_NONE,
                                          names=["abstract_id", "abstract"])
        }

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
            file_extn = os.path.splitext(file_path)[1]
            self._logger.info(f"File extension {file_extn}")
            data_df = self._df_reader_map.get(file_extn, default_df_reader)(file_path)
        # Else read from data frame
        elif isinstance(path_or_dataframe, pd.DataFrame):
            data_df = path_or_dataframe
        else:
            raise ValueError(
                "The type of argument file_path_or_dataframe  must be a str or pandas dataframe, but is {}".format(
                    type(path_or_dataframe)))

        # Filter features
        self._data_df = data_df[
            ["abstract"]]

        # Set up labels
        if "self_label" in data_df.columns:
            self._labels = data_df["self_label"].tolist()
        else:
            self._labels = [False for _ in range(data_df.shape[0])]

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __len__(self):
        return self._data_df.shape[0]

    def __getitem__(self, index):
        x = self._data_df.iloc[index, 0]
        y = self._labels[index]

        if self.transformer:
            x = self.transformer(x)

        if self.label_transformer:
            y = self.label_transformer.map(y)

        return x, y

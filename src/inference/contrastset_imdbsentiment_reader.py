import csv
import logging

import pandas as pd


class ContrastsetImdbSentimentReader:
    """
    https://aclanthology.org/2020.findings-emnlp.117.pdf

    """

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _map_labels(self, l):
        lookup = {
            "negative": "Negative",
            "positive": "Positive"
        }
        return lookup[l.lower()]

    def load_dataset(self, datafile):
        df = pd.read_csv(datafile, delimiter='\t', quotechar='"',
                         quoting=csv.QUOTE_MINIMAL, doublequote=True, header=0,
                         names=["Sentiment", "Text"])
        df["Sentiment"] = df["Sentiment"].apply(self._map_labels)

        return df

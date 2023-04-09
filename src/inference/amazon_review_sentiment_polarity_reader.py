import csv
import logging

import pandas as pd


class AmazonReviewSentimentPolarityReader:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _map_labels(self, l):
        if int(l) == 1:
            return "Negative"
        else:
            return "Positive"

    def load_dataset(self, datafile):
        df = pd.read_csv(datafile, delimiter=',', quotechar='"',
                         quoting=csv.QUOTE_ALL, doublequote=True,
                         names=["Sentiment", "Title", "Text"])
        df["Sentiment"] = df["Sentiment"].apply(self._map_labels)

        return df

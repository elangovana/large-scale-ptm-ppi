import argparse
import csv
import json
import logging
import os
import sys
import tarfile
from typing import Dict

import pandas as pd

from dataset_builder import DatasetBuilder
from inference.amazon_review_sentiment_polarity_reader import AmazonReviewSentimentPolarityReader
from inference.ensemble_predictor import EnsemblePredictor
from locator import Locator


class YelpInference:
    """
    https://huggingface.co/datasets/yelp_polarity/tree/main
    """

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
                         escapechar='\\', quoting=csv.QUOTE_ALL,
                         names=["Sentiment", "Text"])
        df["Sentiment"] = df["Sentiment"].apply(self._map_labels)

        return df


class Inference:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _extract_tar(self, tar_gz_file, dest_dir):
        with  tarfile.open(tar_gz_file) as tf:
            tf.extractall(dest_dir)

    def predict_from_file(self, dataset_reader, data_file, base_artifacts_dir, output_file, batch=32,
                          additional_args=None):
        additional_args = additional_args or {}

        self._logger.info(f"Processing data file {data_file}")

        # if one file in artifacts dir then untar
        self._logger.info(f"Checking if just one tar file exists in {base_artifacts_dir}")
        model_dir = os.path.join(base_artifacts_dir, "model")
        self._extract_tar(os.path.join(base_artifacts_dir, os.listdir(base_artifacts_dir)[0]), model_dir)
        base_artefacts_dir = model_dir
        self._logger.info(f"Setting base dir to {base_artefacts_dir}")

        # Load params
        output_config = os.path.join(base_artefacts_dir, "training_config_parameters.json")
        with open(output_config, "r") as f:
            train_args = json.load(f)

        train_args = {**train_args, **additional_args}

        self._logger.info("Using args :{}".format(train_args))

        # Dataset Builder
        model_factory_name = train_args["modelfactory"]
        df_dataset = dataset_reader.load_dataset(data_file)
        self._logger.info("dataset sets :{}".format(df_dataset["Sentiment"].value_counts()))
        self._logger.info("dataset sample :{}".format(df_dataset.head(n=10)))

        dataset_builder = DatasetBuilder(val_data=df_dataset,
                                         dataset_factory_name=train_args["datasetfactory"],
                                         tokenisor_factory_name=model_factory_name,
                                         batch_size=batch,
                                         addition_args_dict=train_args)

        model_factory = Locator().get(model_factory_name)
        model = model_factory.get_model(dataset_builder.num_classes, checkpoint_dir=base_artefacts_dir, **train_args)

        predictions_data = EnsemblePredictor().predict([model],
                                                       dataset_builder.get_val_dataloader())

        raw_data_iter = df_dataset.to_dict('records')
        self.write_results_to_file(predictions_data, dataset_builder.get_label_mapper(),
                                   output_file,
                                   raw_data_iter)

        self._logger.info(f"Completed file {data_file}")

        return predictions_data

    def write_results_to_file(self, predictions_data_tuple, label_mapper,
                              output_file,
                              raw_data_iter=None):

        result = []

        predictions_tensor = predictions_data_tuple[0]
        confidence_scores = predictions_data_tuple[1]

        if raw_data_iter is None:
            raw_data_iter = [None] * len(predictions_tensor)

        assert len(raw_data_iter) == len(
            predictions_tensor), "The length of raw data iterator {} doesnt match the prediction len {}".format(
            len(raw_data_iter), len(predictions_tensor))

        # Convert indices to labels
        for i, raw_data in enumerate(raw_data_iter):

            pred_i_tensor = predictions_tensor[i]
            conf_i_tensor = confidence_scores[i]

            pred_i = pred_i_tensor.cpu().item()
            conf_i = conf_i_tensor.cpu().tolist()

            label_mapped_confidence = {label_mapper.reverse_map(si): s for si, s in enumerate(conf_i)}
            label_mapped_prediction = label_mapper.reverse_map(pred_i)
            predicted_confidence = conf_i[pred_i]

            r = {
                "prediction": label_mapped_prediction,
                "confidence": predicted_confidence
            }

            r = {**label_mapped_confidence, **r}

            # Add raw data if available
            if isinstance(raw_data, Dict):
                r = {**raw_data, **r}
            else:
                r["raw_data"] = raw_data

            result.append(r)

        self._logger.info("Records to write: {}".format(len(result)))

        if len(result) > 0:
            self._logger.info(f"Writing to file {output_file}")
            # Write json to file
            with open(output_file, "w") as f:
                json.dump(result, f)


def parse_args_run():
    dataset_types = {
        "yelp": YelpInference(),
        "amazon": AmazonReviewSentimentPolarityReader()
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("datajson",
                        help="The json data to predict")
    parser.add_argument("artefactsdir", help="The base of artefacts dir that contains directories of model, vocab etc")
    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("datasettype", help="The type of dataset", choices=list(dataset_types.keys()))

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    args, additional_args = parser.parse_known_args()
    print(args.__dict__)
    # Convert additional args into dict
    additional_dict = {}
    for i in range(0, len(additional_args), 2):
        additional_dict[additional_args[i].lstrip("--")] = additional_args[i + 1]
    print(additional_dict)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    input_file_prefix = os.path.split(args.datajson)[1]
    output_file = os.path.join(args.outdir, f"{args.datasettype}_{input_file_prefix}.json")

    list(Inference().predict_from_file(dataset_types[args.datasettype],
                                       args.datajson, args.artefactsdir, output_file,
                                       args.batch, additional_dict))


if "__main__" == __name__:
    parse_args_run()

import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict

from dataset_builder import DatasetBuilder
from inference.ensemble_predictor import EnsemblePredictor
from locator import Locator


class BatchPredict:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def predict_from_directory(self, datajson, base_artefacts_dir, is_ensemble, output_dir=None, numworkers=None,
                               batch=32, additional_args=None, raw_data_reader_func=None, filter_func=None):
        data_files = [datajson]
        if os.path.isdir(datajson):
            data_files = glob.glob("{}/*.json".format(datajson))

        for d in data_files:
            output_file = "{}.json".format(os.path.join(output_dir, Path(d).name)) if output_dir else None
            self._logger.info("Running inference on file {} with output in {}".format(d, output_file))
            prediction = self.predict_from_file(d, base_artefacts_dir, is_ensemble, output_file, numworkers, batch,
                                                additional_args, raw_data_reader_func, filter_func)

            yield prediction

    def predict_from_file(self, data_file, base_artifacts_dir, is_ensemble, output_file=None, numworkers=None, batch=32,
                          additional_args=None, raw_data_reader_func=None, filter_func=None):
        additional_args = additional_args or {}

        self._logger.info(f"Processing data file {data_file}")

        artifacts_directories = []
        if is_ensemble:
            for d in os.listdir(base_artifacts_dir):
                artifacts_dir = os.path.join(base_artifacts_dir, d)
                artifacts_directories.append(artifacts_dir)
        else:
            artifacts_directories = [base_artifacts_dir]

        # Load params
        output_config = os.path.join(artifacts_directories[0], "training_config_parameters.json")
        with open(output_config, "r") as f:
            train_args = json.load(f)

        train_args = {**train_args, **additional_args}

        self._logger.info("Using args :{}".format(train_args))

        # Dataset Builder
        model_factory_name = train_args["modelfactory"]
        dataset_builder = DatasetBuilder(val_data=data_file, dataset_factory_name=train_args["datasetfactory"],
                                         tokenisor_factory_name=model_factory_name,
                                         num_workers=numworkers, batch_size=batch,
                                         addition_args_dict=train_args)
        # Load ensemble
        models = []
        for artifact_dir in artifacts_directories:
            # Persist params
            output_config = os.path.join(artifacts_directories[0], "training_config_parameters.json")
            with open(output_config, "r") as f:
                train_args = json.load(f)

            model_factory = Locator().get(model_factory_name)
            model = model_factory.get_model(dataset_builder.num_classes, checkpoint_dir=artifact_dir, **train_args)

            models.append(model)

        predictions_data = EnsemblePredictor().predict(models,
                                                       dataset_builder.get_val_dataloader())

        raw_data_iter = raw_data_reader_func(data_file) if raw_data_reader_func else None
        self.write_results_to_file(predictions_data, dataset_builder.get_label_mapper(),
                                   output_file,
                                   raw_data_iter,
                                   filter_func)

        self._logger.info(f"Completed file {data_file}")

        return predictions_data

    def write_results_to_file(self, predictions_data_tuple, label_mapper,
                              output_file,
                              raw_data_iter=None, filter_func=None):

        result = []

        predictions_tensor = predictions_data_tuple[0]
        confidence_scores = predictions_data_tuple[1]
        variation_tensor = predictions_data_tuple[2]
        raw_confidence_scores_tensor = predictions_data_tuple[3]

        # If no filter pass everything through
        default_filter = lambda p, c, s: True
        filter_func = filter_func or default_filter

        if raw_data_iter is None:
            raw_data_iter = [None] * len(predictions_tensor)

        assert len(raw_data_iter) == len(
            predictions_tensor), "The length of raw data iterator {} doesnt match the prediction len {}".format(
            len(raw_data_iter), len(predictions_tensor))

        # Convert indices to labels
        for i, raw_data in enumerate(raw_data_iter):

            pred_i_tensor = predictions_tensor[i]
            conf_i_tensor = confidence_scores[i]
            var_i_tensor = variation_tensor[i]
            raw_conf_i_tensor = raw_confidence_scores_tensor[i]

            pred_i = pred_i_tensor.cpu().item()
            var_i = var_i_tensor.cpu().tolist()
            conf_i = conf_i_tensor.cpu().tolist()
            raw_conf_i = raw_conf_i_tensor.cpu().tolist()

            label_mapped_confidence = {label_mapper.reverse_map(si): s for si, s in enumerate(conf_i)}
            label_mapped_prediction = label_mapper.reverse_map(pred_i)
            predicted_confidence = conf_i[pred_i]
            predicted_confidence_std = var_i[pred_i]

            if not filter_func(label_mapped_prediction, predicted_confidence, predicted_confidence_std): continue

            r = {
                "prediction": label_mapped_prediction,
                "confidence": predicted_confidence,
                "confidence_std": predicted_confidence_std,
                "raw_confidence": raw_conf_i
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

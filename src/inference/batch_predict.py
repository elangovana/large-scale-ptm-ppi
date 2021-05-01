import glob
import json
import logging
import os
from pathlib import Path

from dataset_builder import DatasetBuilder
from inference.ensemble_predictor import EnsemblePredictor
from locator import Locator


class BatchPredict:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def predict_from_directory(self, datajson, base_artefacts_dir, is_ensemble, output_dir=None, numworkers=None,
                               batch=32, additional_args=None):
        data_files = [datajson]
        if os.path.isdir(datajson):
            data_files = glob.glob("{}/*.json".format(datajson))

        for d in data_files:
            output_file = "{}.json".format(os.path.join(output_dir, Path(d).name)) if output_dir else None
            self._logger.info("Running inference on file {} with output in {}".format(d, output_file))
            prediction = self.predict_from_file(d, base_artefacts_dir, is_ensemble, output_file, numworkers, batch,
                                                additional_args)

            yield prediction

    def predict_from_file(self, data_file, base_artifacts_dir, is_ensemble, output_file=None, numworkers=None, batch=32,
                          additional_args=None):
        additional_args = additional_args or {}

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

        predictions, confidence_tensor = EnsemblePredictor().predict(models,
                                                                     dataset_builder.get_val_dataloader())

        self._write_results_to_file(predictions, confidence_tensor, dataset_builder.get_label_mapper(), output_file)

        return predictions, confidence_tensor

    def _write_results_to_file(self, predictions_tensor, confidence_scores_tensor, label_mapper, output_file):
        result = []
        confidence_scores_tensor = confidence_scores_tensor.cpu().tolist()
        predictions = predictions_tensor.cpu().tolist()

        # Convert indices to labels
        for p, scores in zip(predictions, confidence_scores_tensor):
            label_mapped_confidence = {s: label_mapper.reverse_map(si) for si, s in enumerate(scores)}
            label_mapped_predictions = label_mapper.reverse_map(p)
            predicted_confidence = scores[p]

            # Prepare results
            r = {
                "prediction": label_mapped_predictions,
                "confidence": predicted_confidence
            }
            r = {**label_mapped_confidence, **r}

            result.append(r)

        # Write json to file
        with open(output_file, "w") as f:
            json.dump(result, f)

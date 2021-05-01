import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

from dataset_builder import DatasetBuilder
from inference.ensemble_predictor import EnsemblePredictor
from locator import Locator


class BatchPredict:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def predict_from_directory(self, datajson, base_artefacts_dir, output_dir, is_ensemble, numworkers=None, batch=32,
                               additional_args=None):
        data_files = [datajson]
        if os.path.isdir(datajson):
            data_files = glob.glob("{}/*.json".format(datajson))

        for d in data_files:
            self._logger.info("Running inference on file {}".format(d))
            self.predict_from_file(d, base_artefacts_dir, output_dir, is_ensemble, numworkers, batch)

    def predict_from_file(self, data_file, base_artifacts_dir, output_dir, is_ensemble, numworkers=None, batch=32,
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

        print(train_args)

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

        output_file = "{}.json".format(os.path.join(output_dir, Path(data_file).name))
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


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("datajson",
                        help="The json data to predict")

    parser.add_argument("artefactsdir", help="The base of artefacts dir that contains directories of model, vocab etc")
    parser.add_argument("outdir", help="The output dir")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    parser.add_argument("--positives-filter-threshold", help="The threshold to filter positives", type=float,
                        default=0.0)
    parser.add_argument("--numworkers", help="The number of workers to use", type=int, default=None)
    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    parser.add_argument("--ensemble", help="Set to 1 if ensemble model", type=int, default=0, choices={0, 1})
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

    BatchPredict().predict_from_directory(args.datajson, args.artefactsdir, args.outdir, args.ensemble, args.numworkers,
                                          args.batch, additional_dict)

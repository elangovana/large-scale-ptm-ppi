import argparse
import json
import logging
import os
import shutil
import sys
import uuid

from dataset_builder import DatasetBuilder
from train_builder import TrainBuilder

logger = logging.getLogger(__name__)


class TrainPipeline:

    def run_train_k_fold(self, train_dir, val_dir, model_dir, output_dir, kfold_trainprefix=None, checkpoint_dir=None,
                         additional_args=None):

        train_val_objects = []

        # If K Fold prefix is passes in then run k times
        if kfold_trainprefix:
            fold_dirs = os.listdir(train_dir)

            # Prepare the train and val directories
            for f_dir in fold_dirs:
                kfold_dirs = os.listdir("{}{}{}".format(train_dir, os.path.sep, f_dir))
                assert len(
                    kfold_dirs) == 2, "Expecting exactly 2 files or directory under the kfold prefix {} but found {}".format(
                    kfold_trainprefix, kfold_dirs)

                kfold_valprefix = list(set(kfold_dirs) - {kfold_trainprefix})[0]
                fold_train_dir = os.path.join(train_dir, f_dir, kfold_trainprefix)
                fold_val_dir = os.path.join(train_dir, f_dir, kfold_valprefix)

                # Add train val pair
                train_val_objects.append((fold_train_dir, fold_val_dir))

        # Not using K Fold
        else:
            train_val_objects = [(train_dir, val_dir)]

        # Kick off training
        checkpoint_cache = self._load_kfold_check_point(checkpoint_dir) or {}
        results = checkpoint_cache.get("results", [])
        metadata = checkpoint_cache.get("metadata", {})
        checkpoint_completed_folds = [r["data_files"] for r in results]
        for i, (train_o, val_o) in enumerate(
                filter(lambda d: list(d) not in checkpoint_completed_folds, train_val_objects)):
            fold_key = "#".join([train_o, val_o])

            if not fold_key in metadata:
                metadata[fold_key] = {
                    "checkpoint_dir": None
                }

            model_checkpoint_dir = None
            if checkpoint_dir:
                fold_checkpoint_dir = os.path.join(checkpoint_dir, str(uuid.uuid4()))
                metadata[fold_key]["checkpoint_dir"] = metadata[fold_key]["checkpoint_dir"] or fold_checkpoint_dir
                model_checkpoint_dir = metadata[fold_key]["checkpoint_dir"]
                os.makedirs(model_checkpoint_dir, exist_ok=True)

            logger.info("Running fold {} {}".format(i, fold_key))
            # Save before so if training terminates half way through, then the metadata can be loaded
            self._save_kfold_check_point(checkpoint_dir, {"results": results, "metadata": metadata})

            result = self._run_train(train_o, val_o, model_dir, model_checkpoint_dir, additional_args)
            results.append({"data_files": [train_o, val_o]
                               , "result": result
                            })

            # Save at the end of fold..
            self._save_kfold_check_point(checkpoint_dir, {"results": results, "metadata": metadata})

            # Delete  checkpoint for that fold, training complete
            if model_checkpoint_dir:
                shutil.rmtree(model_checkpoint_dir)

        # Write results
        output_results = os.path.join(output_dir, "output.json")
        logger.info(f"Writing output to {output_results}")
        with open(output_results, "w") as f:
            json.dump(results, f)

        self._persist_config(additional_args, model_dir)

        return results

    def _persist_config(self, additional_args, model_dir):
        # Persist params
        output_config = os.path.join(model_dir, "training_config_parameters.json")
        logger.info(f"Writing config to {output_config}")
        with open(output_config, "w") as f:
            json.dump(additional_args, f)

    def _load_kfold_check_point(self, checkpoint_dir):
        if checkpoint_dir is None: return None

        result = None
        checkpoint_point_file = os.path.join(checkpoint_dir, "checkpoint_k_fold.json")
        logger.info("Checking for checkpoints in {}".format(checkpoint_dir))

        if not os.path.exists(checkpoint_point_file): return result

        logger.info("Loading  checkpoints from {}".format(checkpoint_point_file))

        with open(checkpoint_point_file, "r") as f:
            result = json.load(f)

        return result

    def _save_kfold_check_point(self, checkpoint_dir, obj):
        if checkpoint_dir is None: return

        checkpoint_point_file = os.path.join(checkpoint_dir, "checkpoint_k_fold.json")
        logger.info("Saving  fold checkpoint to {}".format(checkpoint_point_file))

        with open(checkpoint_point_file, "w") as f:
            json.dump(obj, f)

    def _run_train(self, train_dir, val_dir, model_dir, checkpointdir, additional_args):
        # Builder
        dataset_builder = DatasetBuilder(val_data=val_dir, dataset_factory_name=additional_args["datasetfactory"],
                                         tokenisor_factory_name=additional_args["modelfactory"], train_data=train_dir,
                                         num_workers=additional_args["numworkers"], batch_size=additional_args["batch"],
                                         addition_args_dict=additional_args)

        train_builder = TrainBuilder(model_factory_name=additional_args["modelfactory"],
                                     scorers=dataset_builder.get_scorers(),
                                     num_classes=dataset_builder.num_classes(),
                                     checkpoint_dir=checkpointdir, epochs=additional_args["epochs"],
                                     grad_accumulation_steps=additional_args.get("gradientaccumulationsteps", 1),
                                     learning_rate=additional_args.get("learningrate", 0.0001),
                                     use_loss_eval=additional_args.get("losseval", 0),
                                     early_stopping_patience=additional_args["earlystoppingpatience"],
                                     model_dir=model_dir,
                                     addition_args_dict=additional_args)

        trainer = train_builder.get_trainer()

        # Run training
        result = trainer.run_train(train_iter=dataset_builder.get_train_dataloader(),
                                   validation_iter=dataset_builder.get_val_dataloader(),
                                   model_network=train_builder.get_network(),
                                   loss_function=train_builder.get_loss_function(),
                                   optimizer=train_builder.get_optimiser(),
                                   pos_label=dataset_builder.positive_label_index()
                                   )
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetfactory",
                        help="The dataset type e.g. datasets.aimed_dataset_factory.AimedDatasetFactory",
                        required=True)

    parser.add_argument("--traindir",
                        help="The input train  dir. If kfoldrootdir is set, then pass the ",
                        default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    parser.add_argument("--valdir",
                        help="The input val dir. If kfoldrootdir is set then this directory is ignored, else mandatory",
                        default=os.environ.get("SM_CHANNEL_VAL", None))

    parser.add_argument("--testdir",
                        help="The input test dir", default=os.environ.get("SM_CHANNEL_TEST", None))

    parser.add_argument("--modelfactory",
                        help="The model factory type e.g. models.bert_model_factory.BertModelFactory",
                        default="models.bert_model_factory.BertModelFactory")

    parser.add_argument("--pretrained_model_dir",
                        help="The pretrained model dir",
                        default=os.environ.get("SM_CHANNEL_PRETRAINED_MODEL", None))

    parser.add_argument("--kfoldtrainprefix",
                        help="If you want to use kFold, set the prefix of the train object. E.g. RootTrainData->OneDirPerFold->kfoldtrainprefix. Objects that do not match the train prefix will be validation  . The number of objects in the root directory will form the number of k",
                        default=None)

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
    parser.add_argument("--modeldir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--checkpointdir", help="The checkpoint dir", default=None)
    parser.add_argument("--checkpointfreq", help="The checkpoint frequency, number of epochs", default=1)

    parser.add_argument("--gradientaccumulationsteps", help="The number of gradient accumulation steps", type=int,
                        default=1)
    parser.add_argument("--learningrate", help="The learningrate", type=float, default=0.0001)

    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--earlystoppingpatience", help="The number of patience epochs epochs", type=int, default=10)
    parser.add_argument("--numworkers", help="The number of workers to use", type=int, default=None)

    parser.add_argument("--uselosseval", help="Whether the best model should be optimised for lowest loss", default=0,
                        choices={0, 1}, type=int)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args, additional = parser.parse_known_args()

    # Convert additional args into dict
    additional_dict = {}
    for i in range(0, len(additional), 2):
        additional_dict[additional[i].lstrip("--")] = additional[i + 1]
    additional_dict["pretrained_model"] = args.pretrained_model_dir

    return args, additional_dict


def main_run():
    args, additional_args = parse_args()
    print(args.__dict__)
    print(additional_args)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Runs
    TrainPipeline().run_train_k_fold(args.traindir,
                                     args.valdir,
                                     kfold_trainprefix=args.kfoldtrainprefix,
                                     checkpoint_dir=args.checkpointdir,
                                     model_dir=args.modeldir,
                                     output_dir=args.outdir,
                                     additional_args={**vars(args), **additional_args}
                                     )


if __name__ == '__main__':
    main_run()

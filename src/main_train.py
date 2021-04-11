import argparse
import json
import logging
import os
import sys

from builder import Builder

logger = logging.getLogger(__name__)


def prepare_run(args, additional_args=None):
    kfold_trainprefix = args.kfoldtrainprefix

    val_dir = args.valdir
    train_dir = args.traindir

    train_val_objects = []

    # If K Fold prefix is passes in then run k times
    if kfold_trainprefix:
        fold_dirs = os.listdir(args.traindir)

        # Set checkpoint as none when running kfold, as the next fold might pick up the old checkpoint before training..
        if args.checkpointdir:
            logger.warning("Checkpoint is not supported during KFold..Disabling checkpointing")
            args.checkpointdir = None

        # Prepare the train and val directories
        for f_dir in fold_dirs:
            kfold_dirs = os.listdir("{}{}{}".format(args.traindir, os.path.sep, f_dir))
            assert len(
                kfold_dirs) == 2, "Expecting exactly 2 files or directory under the kfold prefix {} but found {}".format(
                kfold_trainprefix, kfold_dirs)

            kfold_valprefix = list(set(kfold_dirs) - {kfold_trainprefix})[0]
            train_dir = os.path.join(args.traindir, f_dir, kfold_trainprefix)
            val_dir = os.path.join(args.traindir, f_dir, kfold_valprefix)

            # Add train val pair
            train_val_objects.append((train_dir, val_dir))

    # Not using K Fold
    else:
        train_val_objects = [(train_dir, val_dir)]

    # Kick off training
    results = []
    for i, (train_o, val_o) in enumerate(train_val_objects):
        result = run_train(train_o, val_o, args, additional_args)
        results.append({"id": i
                           , "result": result
                        })

    # Write results
    output_results = os.path.join(args.outdir, "output.json")
    logger.info(f"Writing output to {output_results}")
    with open(output_results, "w") as f:
        json.dump(results, f)


def run_train(train_dir, val_dir, args, additional_args):
    # Builder
    b = Builder(train_data=train_dir, val_data=val_dir,
                dataset_factory_name=args.datasetfactory, model_factory_name=args.modelfactory,
                checkpoint_dir=args.checkpointdir, epochs=args.epochs, grad_accumulation_steps=1,
                num_workers=args.numworkers,
                early_stopping_patience=2, batch_size=args.batch, model_dir=args.modeldir,
                addition_args_dict=additional_args)
    trainer = b.get_trainer()
    # Get data loaders
    train_dataloader, val_dataloader = b.get_train_val_dataloader()
    # Run training
    result = trainer.run_train(train_iter=train_dataloader,
                               validation_iter=val_dataloader,
                               model_network=b.get_network(),
                               loss_function=b.get_loss_function(),
                               optimizer=b.get_optimiser(),
                               pos_label=b.get_pos_label_index()
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

    parser.add_argument("--kfoldtrainprefix",
                        help="If you want to use kFold, set the prefix of the train object. E.g. RootTrainData->OneDirPerFold->kfoldtrainprefix. Objects that do not match the train prefix will be validation  . The number of objects in the root directory will form the number of k",
                        default=None)

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
    parser.add_argument("--modeldir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--checkpointdir", help="The checkpoint dir", default=None)
    parser.add_argument("--checkpointfreq", help="The checkpoint frequency, number of epochs", default=1)

    parser.add_argument("--gradaccumulation", help="The number of gradient accumulation steps", type=int, default=1)
    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--earlystoppingpatience", help="The number of patience epochs epochs", type=int, default=10)
    parser.add_argument("--numworkers", help="The number of workers to use", type=int, default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args, additional = parser.parse_known_args()

    # Convert additional args into dict
    print(additional)
    additional_dict = {}
    for i in range(0, len(additional), 2):
        additional_dict[additional[i].lstrip("--")] = additional[i + 1]

    return args, additional_dict


def main_run():
    args, additional = parse_args()
    print(args.__dict__)
    print(additional)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Runs
    prepare_run(args, additional)


if __name__ == '__main__':
    main_run()

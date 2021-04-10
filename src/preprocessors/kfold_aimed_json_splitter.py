import argparse
import logging
import os
import sys

import pandas as pd

from utils.kfold_wrapper import KFoldWrapper


class KFoldAimedJsonSplitter:

    def __init__(self):
        pass

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def split(self, json_data_file, output_dir, k=10, label_column="interacts", unique_doc_col="documentId"):
        df = pd.read_json(json_data_file)
        k_folds = KFoldWrapper().k_fold(df, label_field_name=label_column, docid_field_name=unique_doc_col, n_splits=k)

        output_files = []
        for i, (train, val) in enumerate(k_folds):
            output_path = os.path.join(output_dir, f"fold_{i}")
            os.makedirs(output_path, exist_ok=True)

            train_file = self._write_file(output_path, train, f"train_{i}")
            val_file = self._write_file(output_path, val, f"validation_{i}")

            output_files.append((train_file, val_file))

        return output_files

    def _write_file(self, output_dir, df, prefix):
        out_file = os.path.join(output_dir, f"{prefix}.json")

        self._logger.info("Writing file {}".format(out_file))
        df.to_json(out_file, orient="records")

        return out_file


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfile",
                        help="The input json Aimed file", required=True)

    parser.add_argument("--outputdir",
                        help="The output directory to write the k folds to", required=True)

    parser.add_argument("--k", help="The number of folds", default=10, type=int)
    parser.add_argument("--kfoldDocId", help="If the folds need to be grouped by this label, e.g. documentId",
                        default=None)
    parser.add_argument("--kfoldLabelColumn", help="The label column to use for stratification", default="interacts")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    KFoldAimedJsonSplitter().split(args.inputfile, args.outputdir, k=args.k, label_column=args.kfoldLabelColumn,
                                   unique_doc_col=args.kfoldDocId)


if __name__ == "__main__":
    run_main()

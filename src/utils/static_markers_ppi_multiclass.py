import argparse
import logging
import sys

import pandas as pd

from datasets.ppi_multiclass_dataset_factory import PpiMulticlassDatasetFactory


class StaticMarkerPpiMulticlass:
    """
    Writes static file with the protein markers
    """

    def create(self, ppi_multiclass_file, output_file, additional_cols=None):
        dataset_factory = PpiMulticlassDatasetFactory()
        dataset = dataset_factory.get_dataset(ppi_multiclass_file)
        label_mapper = dataset.label_transformer
        dataset.label_transformer = None

        data = [{"x": x, "y": label_mapper.map(y), "y_raw": y} for x, y in dataset]

        df = pd.DataFrame(data)

        if additional_cols:
            cols_to_copy = [c.strip(" ") for c in additional_cols.split(",")]
            raw_input_json = pd.read_json(ppi_multiclass_file)
            for c in cols_to_copy:
                df[c] = raw_input_json[c]

        df.to_json(output_file, orient="records")


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfile",
                        help="The input ppi multiclass file", required=True)

    parser.add_argument("--outputfile",
                        help="The output json file", required=True)

    parser.add_argument("--additionalcols",
                        help="The additional columns, comma separated, to copy to output", required=False, default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    StaticMarkerPpiMulticlass().create(args.inputfile, args.outputfile, args.additionalcols)


if __name__ == "__main__":
    run_main()

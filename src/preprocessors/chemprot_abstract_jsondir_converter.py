import argparse
import logging
import os
import sys

from preprocessors.chemprot_abstract_json_converter import ChemprotAbstractJsonConverter


class ChemprotAbstractJsonDirConverter:
    """
    Converts a directory of chem prot files into json https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/, with abstract intact
    """

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def convert(self, inputdir: str, abstract_file_suffix: str, entities_file_suffix: str, output_dir: str):
        file_converter = ChemprotAbstractJsonConverter()
        os.makedirs(output_dir, exist_ok=True)
        for entities_file_name in filter(lambda x: x.endswith(entities_file_suffix), os.listdir(inputdir)):
            matching_filename_prefix = entities_file_name[:-len(entities_file_suffix)]
            abstract_file_name = "{}{}".format(matching_filename_prefix, abstract_file_suffix)
            abstract_file = os.path.join(inputdir, abstract_file_name)
            entities_file = os.path.join(inputdir, entities_file_name)
            dest_json_file_name = matching_filename_prefix + ".rel.json"
            dest_json_file = os.path.join(output_dir, dest_json_file_name)
            if os.path.exists(abstract_file):
                self._logger.info(
                    "Processing file set {}, with output {}".format((abstract_file, entities_file), dest_json_file))
                file_converter.convert(abstract_file, entities_file, dest_json_file=dest_json_file)
            else:
                self._logger.warning(
                    f"The corresponding abstract {abstract_file} not found for file {entities_file_name}")


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputdir",
                        help="The input directory containing the files ", required=True)

    parser.add_argument("--outputdir",
                        help="The output directory", required=True)
    parser.add_argument("--abstractfilesuffix",
                        help="The suffix of the abstract file", required=False, default=".abstract.tsv")
    parser.add_argument("--entitiesfilesuffix",
                        help="The suffix of the abstract file", required=False, default=".anon.txt")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    ChemprotAbstractJsonDirConverter().convert(args.inputdir, args.abstractfilesuffix, args.entitiesfilesuffix,
                                               args.outputdir)


if __name__ == "__main__":
    run_main()

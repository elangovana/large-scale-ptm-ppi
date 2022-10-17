import argparse
import logging
import os
import sys
import tarfile

import pandas as pd

from inference.batch_predict import BatchPredict


class ChemprotBatchPredict:

    def __init__(self, batch_predict, filter_std=0.15, use_filter=False, filter_threshold_negative=None):
        self.filter_threshold_negative = filter_threshold_negative
        self.use_filter = use_filter
        self.filter_std = filter_std
        self.batch_predict = batch_predict

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _get_chemprot_inference_reader(self, data_File):
        return pd.read_json(data_File).to_dict(orient="record")

    def _extract_tar(self, tar_gz_file, dest_dir):
        with  tarfile.open(tar_gz_file) as tf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, dest_dir)

    def predict_from_dir(self, datajson, base_artefacts_dir, is_ensemble, output_dir=None, numworkers=None,
                         batch=32, additional_args=None, raw_data_reader_func=None, filter_func=None):
        # if one file in artifacts dir then untar
        self._logger.info(f"Checking if just one tar file exists in {base_artefacts_dir}")

        if len(os.listdir(base_artefacts_dir)) == 1 and os.listdir(base_artefacts_dir)[0].endswith("tar.gz"):
            model_dir = os.path.join(base_artefacts_dir, "model")
            self._extract_tar(os.path.join(base_artefacts_dir, os.listdir(base_artefacts_dir)[0]), model_dir)
            base_artefacts_dir = model_dir
            self._logger.info(f"Setting base dir to {base_artefacts_dir}")

        raw_data_reader_func = raw_data_reader_func or self._get_chemprot_inference_reader

        # Use filter if
        default_filter = lambda p, c, s: True
        if self.use_filter:
            self._logger.info("Using filter with threshold {}".format(self.filter_std))
            default_filter = lambda p, c, s: (c > self.filter_threshold_negative and p == 'NEGATIVE') or (
                    s <= self.filter_std)
        filter_func = filter_func or default_filter

        return list(self.batch_predict.predict_from_directory(datajson, base_artefacts_dir,
                                                              is_ensemble, output_dir=output_dir,
                                                              numworkers=numworkers, batch=batch,
                                                              additional_args=additional_args,
                                                              raw_data_reader_func=raw_data_reader_func,
                                                              filter_func=filter_func))

    def predict_from_file(self, datajson, base_artefacts_dir, is_ensemble, output_file, numworkers=None, batch=32,
                          additional_args=None, raw_data_reader_func=None, filter_func=None):
        raw_data_reader_func = raw_data_reader_func or self._get_chemprot_inference_reader

        # if one file in artifacts dir then untar
        self._logger.info(f"Checking if just one tar file exists in {base_artefacts_dir}")
        if len(os.listdir(base_artefacts_dir)) == 1 and os.listdir(base_artefacts_dir)[0].endswith("tar.gz"):
            model_dir = os.path.join(base_artefacts_dir, "model")
            self._extract_tar(os.path.join(base_artefacts_dir, os.listdir(base_artefacts_dir)[0]), model_dir)
            base_artefacts_dir = model_dir
            self._logger.info(f"Setting base dir to {base_artefacts_dir}")

        return self.batch_predict.predict_from_file(datajson, base_artefacts_dir,
                                                    is_ensemble, output_file=output_file,
                                                    numworkers=numworkers, batch=batch,
                                                    additional_args=additional_args,
                                                    raw_data_reader_func=raw_data_reader_func,
                                                    filter_func=filter_func
                                                    )


def parse_args_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir",
                        help="The json data dir to predict")
    parser.add_argument("artefactsdir", help="The base of artefacts dir that contains directories of model, vocab etc")
    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    parser.add_argument("--numworkers", help="The number of workers to use", type=int, default=None)
    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    parser.add_argument("--ensemble", help="Set to 1 if ensemble model", type=int, default=0, choices={0, 1})
    parser.add_argument("--filter", help="Set to 1 if ensemble model", type=int, default=0, choices={0, 1})
    parser.add_argument("--filterstdthreshold", help="Set the filter threshold", type=float, default=1.0)

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

    ChemprotBatchPredict(BatchPredict(), use_filter=args.filter,
                         filter_std=args.filterstdthreshold).predict_from_dir(args.datadir,
                                                                              args.artefactsdir,
                                                                              args.ensemble,
                                                                              args.outdir,
                                                                              args.numworkers,
                                                                              args.batch,
                                                                              additional_dict)


if "__main__" == __name__:
    parse_args_run()

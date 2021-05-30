import argparse
import logging
import sys

import pandas as pd

from inference.batch_predict import BatchPredict


class PpiMulticlassBatchPredict:

    def __init__(self, batch_predict, filter_std=0.15, use_filter=False):
        self.use_filter = use_filter
        self.filter_std = filter_std
        self.batch_predict = batch_predict

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _get_ppi_multiclass_inference_reader(self, data_File):
        return pd.read_json(data_File).to_dict(orient="record")

    def predict_from_directory(self, datajson, base_artefacts_dir, is_ensemble, output_dir, numworkers=None, batch=32,
                               additional_args=None, raw_data_reader_func=None, filter_func=None):
        raw_data_reader_func = raw_data_reader_func or self._get_ppi_multiclass_inference_reader

        # Use filter if
        default_filter = lambda p, c, s: True
        if self.use_filter:
            default_filter = lambda p, c, s: p != 'other'
        filter_func = filter_func or default_filter

        # Invoke underlying batch predict
        return list(self.batch_predict.predict_from_directory(datajson, base_artefacts_dir,
                                                              is_ensemble, output_dir=output_dir,
                                                              numworkers=numworkers, batch=batch,
                                                              additional_args=additional_args,
                                                              raw_data_reader_func=raw_data_reader_func,
                                                              filter_func=filter_func))

    def predict_from_file(self, datajson, base_artefacts_dir, is_ensemble, output_file, numworkers=None, batch=32,
                          additional_args=None, raw_data_reader_func=None, filter_func=None):
        raw_data_reader_func = raw_data_reader_func or self._get_ppi_multiclass_inference_reader
        return self.batch_predict.predict_from_file(datajson, base_artefacts_dir,
                                                    is_ensemble, output_file=output_file,
                                                    numworkers=numworkers, batch=batch,
                                                    additional_args=additional_args,
                                                    raw_data_reader_func=raw_data_reader_func,
                                                    filter_func=filter_func
                                                    )


def parse_args_run():
    global args, additional_dict
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
    parser.add_argument("--filter", help="Set to 1 if ensemble model", type=int, default=0, choices={0, 1})

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

    PpiMulticlassBatchPredict(BatchPredict(), use_filter=args.filter).predict_from_directory(args.datajson,
                                                                                             args.artefactsdir,
                                                                                             args.ensemble,
                                                                                             args.outdir,
                                                                                             args.numworkers,
                                                                                             args.batch,
                                                                                             additional_dict)


if "__main__" == __name__:
    parse_args_run()

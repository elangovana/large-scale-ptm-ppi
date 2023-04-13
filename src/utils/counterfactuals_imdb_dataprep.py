#!/usr/bin/env python
import argparse
import json
import logging
import os.path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.diff_sentences import DiffSentences

TOTAL_SIZE = 3400

POS_RATE = 0.5
ADV_THRESHOLD = 0.25
AFF_THRESHOLD = 0.50


class CounterfactualsImdbDataPrep:

    def _compute_affability_rate(self, samples, threshold):
        aff_diff = DiffSentences().pairwise_edit_distance_ratio(list(samples), list(samples), score_cutoff=0)

        # Set max distance in diagonals so that it doesn't come up as the closest match
        for i in range(len(aff_diff)):
            aff_diff[i, i] = 1000

        closest_match = np.min(aff_diff, axis=-1)
        closest_match_idx = np.argmin(aff_diff, axis=-1)
        aff_diff_min = [1 for i in closest_match if i < threshold]
        aff_rate = sum(aff_diff_min) / len(samples)

        debug_df = pd.DataFrame(samples)
        debug_df["closest_dist"] = closest_match
        debug_df["closest_match"] = [list(samples)[i] for i in closest_match_idx]

        return aff_rate, debug_df

    def _compute_adv_rate(self, samples1, samples2, threshold):
        adv_diff = DiffSentences().pairwise_edit_distance_ratio(list(samples1), list(samples2), score_cutoff=0)

        closest_match = np.min(adv_diff, axis=-1)
        closest_match_idx = np.argmin(adv_diff, axis=-1)

        # Adv P -> N debug info
        debug_df = pd.DataFrame(samples1)
        debug_df["closest_dist"] = closest_match
        debug_df["closest_match"] = [list(samples2)[i] for i in closest_match_idx]

        adv_diff_min = [1 for i in closest_match if i < threshold]
        adv_rate = sum(adv_diff_min) / len(samples1)

        return adv_rate, debug_df

    def get_stats(self, df, threshold_adv=.25, threshold_aff=.50):
        num_unique = df["Text"].nunique()
        pos_ratio = len(df.query("Sentiment == 'Positive'")) / len(df)
        pos_texts = df.query("Sentiment == 'Positive'")["Text"]
        neg_texts = df.query("Sentiment == 'Negative'")["Text"]

        pos_adv_rate, adv_debug = self._compute_adv_rate(pos_texts, neg_texts, threshold_adv)
        pos_aff_rate, aff_pos_debug = self._compute_affability_rate(pos_texts, threshold_aff)

        neg_aff_rate, neg_pos_debug = self._compute_affability_rate(neg_texts, threshold_aff)

        result = {
            "Unique": num_unique,
            "PosRate": pos_ratio,
            "AdvRatePN": pos_adv_rate,
            "AffRateP": pos_aff_rate,
            "AffRateN": neg_aff_rate,
            "Total": len(df)
        }

        debug = {
            "adv_p_n": adv_debug,
            "aff_p": aff_pos_debug,
            "aff_n": neg_pos_debug
        }

        return result, debug

    def _add_col_batch_id(self, df, df_counter_fact):
        dict_text_index = {}
        dict_batch_index = {}
        for r in df_counter_fact.to_dict('records'):
            dict_text_index[r["Text"]] = r["batch_id"]
            if r["batch_id"] not in dict_batch_index: dict_batch_index[r["batch_id"]] = {}

            counter_label = 'Negative' if r["Sentiment"] == 'Positive' else 'Positive'
            dict_batch_index[r["batch_id"]][counter_label] = r["Text"]

        df["batch_id"] = df["Text"].apply(lambda x: dict_text_index.get(x, -1)).astype(int)
        df["counter_text"] = df.apply(lambda x: dict_batch_index.get(x["batch_id"], {}).get(x["Sentiment"], None),
                                      axis=1)
        df["counter_sentiment"] = df.apply(lambda x: ('Negative' if x["Sentiment"] == 'Positive' else 'Positive')
        if x["counter_text"] else None, axis=1)

        return df

    def prep_counterfactual(self, df, adv_rate_pos=0.1, target_total_size=3400):

        target_neg_samples = min(int(target_total_size * (1 - POS_RATE)),
                                 len(df.query("Sentiment == 'Negative'")))

        target_pos_samples = target_total_size - target_neg_samples
        target_adv_neg_samples = int(adv_rate_pos * target_pos_samples)
        target_non_adv_neg_samples = target_neg_samples - target_adv_neg_samples

        self._logger.info(
            f"Target Total size:{target_total_size},  target_neg_samples:{target_neg_samples} target_pos_samples{target_pos_samples}")
        self._logger.info(f"DF value counts {df['Sentiment'].value_counts()}, Total: {len(df)},")
        self._logger.info(
            f"target_non_adv_neg_samples:{target_non_adv_neg_samples} target_adv_neg_samples{target_adv_neg_samples}")

        df_pos_with_cf = df.query("batch_id != -1 and Sentiment == 'Positive'")
        df_pos_without_cf = df.query("batch_id == -1 and Sentiment == 'Positive'")
        self._logger.info(f"Positive samples: with CF :{len(df_pos_with_cf)}, without CF :{len(df_pos_without_cf)}")

        df_neg_with_cf = df.query("batch_id != -1 and Sentiment == 'Negative'")
        df_neg_without_cf = df.query("batch_id == -1 and Sentiment == 'Negative'")
        self._logger.info(f"Negative samples: with CF :{len(df_neg_with_cf)}, without CF :{len(df_neg_without_cf)}")

        # Get n positives
        pos_samp_with_adv_size = min(len(df_pos_with_cf), target_pos_samples)
        df_pos_with_adv = df_pos_with_cf.sample(n=pos_samp_with_adv_size)
        df_pos_without_adv = df_pos_without_cf[["Sentiment", "Text", "batch_id"]].sample(
            n=target_pos_samples - pos_samp_with_adv_size)

        # Get paired n negatives
        df_paired = df_pos_with_adv.sample(n=target_adv_neg_samples)
        df_adv_neg = df_paired[["counter_sentiment", "counter_text", "batch_id"]]
        df_adv_neg.columns = ["Sentiment", "Text", "batch_id"]

        # Random neg
        df_neg_random = df_neg_without_cf[["Sentiment", "Text", "batch_id"]].sample(n=target_non_adv_neg_samples).copy()

        return pd.concat([df_adv_neg, df_pos_with_adv[["Sentiment", "Text", "batch_id"]],
                          df_pos_without_adv, df_neg_random])

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def run(self, output_dir):
        train_counterfacts_data_url = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/combined/paired/train_paired.tsv"
        val_counterfacts_data_url = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/combined/paired/dev_paired.tsv"

        # train_orig_data_url = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/combined/train.tsv"
        # df_train_orig = pd.read_csv(train_orig_data_url, sep="\t")
        # val_orig_data_url = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/combined/dev.tsv"
        # df_val_orig = pd.read_csv(val_orig_data_url, sep="\t")

        train_orig_data_url = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/orig/eighty_percent/train.tsv"
        df_train_orig = pd.read_csv(train_orig_data_url, sep="\t")
        df_train_orig, df_val_orig = train_test_split(df_train_orig, test_size=0.2, random_state=42)

        self._logger.info(f"Train, val: {df_train_orig.shape, df_val_orig.shape}")

        df_counterfacts_train = pd.read_csv(train_counterfacts_data_url, sep="\t")
        self._logger.info(f"Counter factual train: {df_counterfacts_train.shape}")

        df_counterfacts_val = pd.read_csv(val_counterfacts_data_url, sep="\t")
        self._logger.info(f"Counter factual val: {df_counterfacts_val.shape}")

        df_train = df_train_orig.pipe(self._add_col_batch_id, df_counterfacts_train)
        df_val = df_val_orig.pipe(self._add_col_batch_id, df_counterfacts_val)

        self._logger.info(f"Original train, val: {df_train.shape, df_val.shape}")

        counterfact_train_stats, counterfact_train_stats_debug = self.get_stats(df_counterfacts_train)
        self.dump_json(counterfact_train_stats, os.path.join(output_dir, "counterfact_train_stats.json"))
        self.dump_json(counterfact_train_stats, os.path.join(output_dir, "counterfact_train_stats_debug.json"))

        adv_ranges = [0, 0.10, 0.20, .30, 0.90]

        for adv_rate in adv_ranges:
            for i in range(5):
                df_train_prepared = self.prep_counterfactual(df_train, adv_rate_pos=adv_rate,
                                                             target_total_size=TOTAL_SIZE)

                result_train_stats, train_debug_stats = self.get_stats(df_train_prepared, threshold_adv=ADV_THRESHOLD,
                                                                       threshold_aff=AFF_THRESHOLD)
                self._logger.info(json.dumps(result_train_stats))
                prefix_path = "{:02d}_{:02d}_{:02d}_{:02d}".format(int(result_train_stats["AdvRatePN"] * 100),
                                                                   int(result_train_stats["AffRateP"] * 100),
                                                                   int(result_train_stats["AffRateN"] * 100),
                                                                   i + 1
                                                                   )

                self.dump_json(result_train_stats, os.path.join(output_dir, f"stats_{prefix_path}_train.json"))
                self.dump_json(train_debug_stats, os.path.join(output_dir, f"debug_stats_{prefix_path}_val.json"))

                df_val_prepared = self.prep_counterfactual(df_val, adv_rate_pos=adv_rate,
                                                           target_total_size=int(TOTAL_SIZE * 0.2))
                result_val_stats, val_debug_stats = self.get_stats(
                    df_val_prepared, threshold_adv=ADV_THRESHOLD, threshold_aff=AFF_THRESHOLD)
                self._logger.info(json.dumps(result_val_stats))

                self.dump_json(result_val_stats, os.path.join(output_dir, f"stats_{prefix_path}_val.json"))
                self.dump_json(val_debug_stats, os.path.join(output_dir, f"debug_stats_{prefix_path}_val.json"))

                df_train_prepared.reset_index().to_json(os.path.join(output_dir, prefix_path, "train.json"))
                df_val_prepared.reset_index().to_json(os.path.join(output_dir, prefix_path, "val.json"))

    def dump_json(self, obj, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(obj, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outputdir",
                        help="The output dir", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    return args


def main_run():
    args = parse_args()
    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Runs
    CounterfactualsImdbDataPrep().run(args.outputdir)


if __name__ == '__main__':
    main_run()

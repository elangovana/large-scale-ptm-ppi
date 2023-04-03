import argparse
import csv
import logging
import sys

from utils.diff_sentences import DiffSentences


class DiffDatasetCounterFactuals:
    """
     https://github.com/acmi-lab/counterfactually-augmented-data/blob/master/sentiment/combined/paired/train_paired.tsv
    """

    def compare(self, tsv_file_or_handle):
        result = []
        data = self._load_file_or_handle(tsv_file_or_handle)
        sent_word_differ = DiffSentences()
        # Compare the counter factual data
        for b, b_items in data.items():
            assert len(b_items) == 2, f"Expecting only 2 items per batch, found {len(b_items)} in id {b}"
            assert b_items[0]["label"] != b_items[1]["label"], "Expecting batch to have different labels in id {b}"
            # Always ensure that label 0 is negative and 1 is positive
            b_items.sort(key=lambda x: x["label"])

            word_diff = sent_word_differ.get_edit_span(b_items[0]["text"], b_items[1]["text"])
            result.append(word_diff)
        return result

    def _load_file_or_handle(self, tsv_file_or_handle):
        if isinstance(tsv_file_or_handle, str):
            with open(tsv_file_or_handle) as f:
                result = self._load_from_handle(f)
        else:
            result = self._load_from_handle(tsv_file_or_handle)

        return result

    def _load_from_handle(self, f):
        result = {}
        csv_reader = csv.reader(f, delimiter="\t")
        # skip header
        next(csv_reader)

        for l in csv_reader:
            batch_id = l[2]
            if batch_id not in result: result[batch_id] = []
            result[batch_id].append({
                "label": l[0],
                "text": l[1],
                "batchid": batch_id
            })
        return result


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("inputfile",
                        help="The input counterfactuals sample data")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    result = DiffDatasetCounterFactuals().compare(args.inputfile)
    print(result)


if __name__ == "__main__":
    run_main()

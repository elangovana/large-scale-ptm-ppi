import logging

from sklearn.model_selection import StratifiedKFold


class KFoldWrapper:
    """
    KFold wrapper takes into account additional conditions such as column key uniqueness
    """

    def k_fold(self, df, label_field_name, docid_field_name=None, n_splits=10):
        if docid_field_name is None:
            yield from self._k_fold_ignore_doc(df, label_field_name, n_splits)
        else:
            yield from self._k_fold_unique_doc(df, label_field_name, docid_field_name, n_splits)

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @staticmethod
    def _get_label_distribution(df, label_field_name):
        label_counts_raw = df[label_field_name].value_counts()
        label_counts_percentage = label_counts_raw * 100 / sum(label_counts_raw.values)

        return label_counts_percentage

    def _k_fold_unique_doc(self, df, label_field_name, docid_field_name, n_splits=10, random_state=777):
        """
        Split taking into account a key column
        """
        self._logger.info("Splitting such that the {} is unique across datasets".format(docid_field_name))
        kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        unique_docids = df[docid_field_name].unique()

        # Do a approx so that the labels are somewhat stratified in the split,
        # i.e.get the most common label for a given dodc id
        approx_y = [df.query("{} == '{}'".format(docid_field_name, p))[label_field_name].value_counts().idxmax() for p
                    in unique_docids]
        for train_index, test_index in kf.split(unique_docids, approx_y):
            train_doc, test_doc = unique_docids[train_index], unique_docids[test_index]
            train = df[df[docid_field_name].isin(train_doc)]
            val = df[df[docid_field_name].isin(test_doc)]

            self._log_label_distribution(label_field_name, train, val)
            yield train, val

    def _k_fold_ignore_doc(self, df, label_field_name, n_splits=10, random_state=777):
        """
        Regular stratified split
        :param df:
        :param label_field_name:
        :param n_splits:
        :param random_state:
        :return:
        """
        kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

        for train_index, test_index in kf.split(df, df[label_field_name]):
            train, val = df.iloc[train_index], df.iloc[test_index]

            self._log_label_distribution(label_field_name, train, val)

            yield train, val

    def _log_label_distribution(self, label_field_name, train, val):
        """
        Logs label distribution in train and val
        :param label_field_name:
        :param train:
        :param val:
        :return:
        """
        self._logger.info("Train split label distribution {} ".format(
            str(self._get_label_distribution(train, label_field_name)).replace("\n", "\t")))
        self._logger.info("Validation split label distribution {} ".format(
            str(self._get_label_distribution(val, label_field_name)).replace("\n", "\t")))

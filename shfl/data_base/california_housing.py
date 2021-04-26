import sklearn.datasets

from shfl.data_base import data_base as db


class CaliforniaHousing(db.DataBase):
    """Loads the California housing dataset.

    # References:
    [California housing dataset](https://scikit-learn.org/stable/modules/generated/
        sklearn.datasets.fetch_california_housing.html)
    """
    def load_data(self):
        """Loads the California housing dataset.
        """
        all_data = sklearn.datasets.fetch_california_housing()
        data = all_data["data"]
        labels = all_data["target"]

        if self._shuffle:
            data, labels = db.shuffle_rows(data, labels)

        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = \
            db.split_train_test(data, labels)

        return self.data

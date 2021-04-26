import sklearn.datasets
from shfl.data_base import data_base as db


class Iris(db.DataBase):
    """Loads the Iris dataset.

    # References:
    [Iris dataset](https://scikit-learn.org/stable/modules/generated/
        sklearn.datasets.load_iris.html)
    """
    def load_data(self):
        all_data = sklearn.datasets.load_iris()
        data = all_data["data"]
        labels = all_data["target"]

        if self._shuffle:
            data, labels = db.shuffle_rows(data, labels)

        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = \
            db.split_train_test(data, labels)

        return self.data

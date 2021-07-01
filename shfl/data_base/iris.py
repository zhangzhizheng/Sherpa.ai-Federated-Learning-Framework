import sklearn.datasets

from shfl.data_base.data_base import LabeledDatabase


class Iris(LabeledDatabase):
    """Loads the Iris dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
    [Iris dataset](https://scikit-learn.org/stable/modules/generated/
        sklearn.datasets.load_iris.html)
    """

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self, train_proportion=0.8, shuffle=True):
        """Loads the train and test data.

        # Arguments:
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).
        shuffle: Optional; Boolean for shuffling rows before the
            train/test split (default is True).
        """

        if self._data is None or self._labels is None:
            all_data = sklearn.datasets.load_iris()
            self._data = all_data["data"]
            self._labels = all_data["target"]

        self.split_data(train_proportion, shuffle)

        return self.data

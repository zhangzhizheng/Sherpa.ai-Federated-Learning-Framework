import sklearn.datasets

from shfl.data_base.data_base import LabeledDatabase


class Iris(LabeledDatabase):
    """Loads the Iris dataset.

    Implements base class [LabeledDataBase](./#labeleddatabase-class).

    # Arguments:
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).
        shuffle: Optional; Boolean for shuffling rows before the
            train/test split (default is True).

    # References:
    [Iris dataset](https://scikit-learn.org/stable/modules/generated/
        sklearn.datasets.load_iris.html)
    """

    def __init__(self, train_proportion=0.8, shuffle=True):
        all_data = sklearn.datasets.load_iris()
        data = all_data["data"]
        labels = all_data["target"]
        super().__init__(data, labels, train_proportion, shuffle)

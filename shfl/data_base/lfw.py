import sklearn.datasets
# Tensorflow warning
# pylint: disable=no-name-in-module
from tensorflow.keras.utils import to_categorical

from shfl.data_base.data_base import LabeledDatabase


class Lfw(LabeledDatabase):
    """Loads the LFW dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
        [Labeled Faces in the Wild dataset](https://scikit-learn.org/stable/
            datasets/index.html#labeled-faces-in-the-wild-dataset)
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
            all_data = sklearn.datasets.fetch_lfw_people(color=True)
            self._data = all_data["images"]
            self._labels = to_categorical(all_data["target"])

        self.split_data(train_proportion, shuffle)

        return self.data

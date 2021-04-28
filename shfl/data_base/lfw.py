import sklearn.datasets
from tensorflow.keras.utils import to_categorical

from shfl.data_base.data_base import LabeledDatabase


class Lfw(LabeledDatabase):
    """Loads the LFW dataset.

    Implements base class [LabeledDataBase](./#labeleddatabase-class).

    # Arguments:
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).
        shuffle: Optional; Boolean for shuffling rows before the
            train/test split (default is True).

    # References:
        [Labeled Faces in the Wild dataset](https://scikit-learn.org/stable/
            datasets/index.html#labeled-faces-in-the-wild-dataset)
    """
    def __init__(self, train_proportion=0.8, shuffle=True):
        all_data = sklearn.datasets.fetch_lfw_people(color=True)
        data = all_data["images"]
        labels = to_categorical(all_data["target"])
        super().__init__(data, labels, train_proportion, shuffle)

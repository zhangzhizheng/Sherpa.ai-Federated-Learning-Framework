from sklearn.datasets import fetch_lfw_people
from tensorflow.keras.utils import to_categorical

from shfl.data_base import data_base as db


class Lfw(db.DataBase):
    """Loads the LFW dataset.

    # References:
        [Labeled Faces in the Wild dataset](https://scikit-learn.org/stable/
            datasets/index.html#labeled-faces-in-the-wild-dataset)
    """

    def load_data(self):
        all_data = fetch_lfw_people(color=True)
        data = all_data["images"]
        labels = to_categorical(all_data["target"])

        if self._shuffle:
            data, labels = db.shuffle_rows(data, labels)

        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = \
            db.split_train_test(data, labels)

        return self.data

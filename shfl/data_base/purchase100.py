from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.utils import to_categorical
import numpy as np

from shfl.data_base import data_base as db


class Purchase100(db.DataBase):
    """Loads the Purchase100 dataset.

    # References:
    [Purchase100 dataset](https://www.kaggle.com/c/
        acquire-valued-shoppers-challenge).
    """

    def load_data(self):
        path_features = get_file(
            "purchase100",
            origin="https://github.com/xehartnort/Purchase100-dataset/releases/download/v1.1/purchase100.npz",
            extract=True,
            file_hash="0d7538b9806e7ee622e1a252585e7768",  # md5 hash
            cache_dir='~/.sherpa-ai')

        all_data = np.load(path_features)
        data = all_data['features']
        labels = to_categorical(all_data['labels'])

        if self._shuffle:
            data, labels = db.shuffle_rows(data, labels)

        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = db.split_train_test(
                data, labels)

        return self.data

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.utils import to_categorical
import numpy as np

from shfl.data_base.data_base import LabeledDatabase


class Purchase100(LabeledDatabase):
    """Loads the Purchase100 dataset.

    Implements base class [LabeledDataBase](./#labeleddatabase-class).

    # Arguments:
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).
        shuffle: Optional; Boolean for shuffling rows before the
            train/test split (default is True).

    # References:
    [Purchase100 dataset](https://www.kaggle.com/c/
        acquire-valued-shoppers-challenge).
    """

    def __init__(self, train_proportion=0.8, shuffle=True):
        path_features = get_file(
            "purchase100",
            origin="https://github.com/xehartnort/Purchase100-dataset/"
                   "releases/download/v1.1/purchase100.npz",
            extract=True,
            file_hash="0d7538b9806e7ee622e1a252585e7768",  # md5 hash
            cache_dir='~/.sherpa-ai')

        all_data = np.load(path_features)
        data = all_data['features']
        labels = to_categorical(all_data['labels'])
        super().__init__(data, labels, train_proportion, shuffle)

"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

# Tensorflow warning
# pylint: disable=no-name-in-module
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.utils import to_categorical
import numpy as np

from shfl.data_base.data_base import LabeledDatabase


class Purchase100(LabeledDatabase):
    """Loads the Purchase100 dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
    [Purchase100 dataset](https://www.kaggle.com/c/
        acquire-valued-shoppers-challenge).
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
            path_features = get_file(
                "purchase100",
                origin="https://github.com/xehartnort/Purchase100-dataset/"
                       "releases/download/v1.1/purchase100.npz",
                extract=True,
                file_hash="0d7538b9806e7ee622e1a252585e7768",  # md5 hash
                cache_dir='~/.sherpa-ai')

            all_data = np.load(path_features)
            self._data = all_data['features']
            self._labels = to_categorical(all_data['labels'])

        self.split_data(train_proportion, shuffle)

        return self.data

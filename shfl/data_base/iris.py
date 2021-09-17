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

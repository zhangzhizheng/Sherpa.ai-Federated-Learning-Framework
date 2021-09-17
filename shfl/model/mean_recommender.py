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

import numpy as np

from shfl.model.recommender import Recommender


class MeanRecommender(Recommender):
    """Mean recommender model.

    Implements the class [Recommender](./#recommender).

    Given a set of labels (i.e. ratings) in the training set of each client,
    it uses their mean value to make predictions.
    """

    def __init__(self):
        super().__init__()
        self._mu = None

    def train_recommender(self, data, labels, **kwargs):
        """See base class."""
        self._mu = np.mean(labels)

    def predict_recommender(self, data):
        """See base class."""
        predictions = np.full(len(data), self._mu)
        return predictions

    def get_model_params(self):
        """See base class."""
        return self._mu

    def set_model_params(self, params):
        """See base class."""
        self._mu = params

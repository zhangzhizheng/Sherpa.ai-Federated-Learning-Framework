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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shfl.model.deep_learning_model import DeepLearningModel
from shfl.model.model import TrainableModel
from shfl.model.kmeans_model import KMeansModel
from shfl.model.linear_regression_model import LinearRegressionModel
from shfl.model.linear_classifier_model import LinearClassifierModel
from shfl.model.recommender import Recommender
from shfl.model.mean_recommender import MeanRecommender
from shfl.model.content_based_recommender import ContentBasedRecommender
from shfl.model.deep_learning_model_pt import DeepLearningModelPyTorch
from shfl.model.vertical_deep_learning_model import VerticalNeuralNetClientModel
from shfl.model.vertical_deep_learning_model import VerticalNeuralNetServerModel

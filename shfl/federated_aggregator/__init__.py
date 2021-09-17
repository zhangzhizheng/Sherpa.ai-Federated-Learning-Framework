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

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.federated_aggregator.weighted_fedavg_aggregator import WeightedFedAggregator
from shfl.federated_aggregator.fedsum_aggregator import FedSumAggregator
from shfl.federated_aggregator.iowa_federated_aggregator import IowaFederatedAggregator
from shfl.federated_aggregator.cluster_fedavg_aggregator import cluster_fed_avg_aggregator
from shfl.federated_aggregator.norm_clip_aggregators import NormClipAggregator
from shfl.federated_aggregator.norm_clip_aggregators import CDPAggregator
from shfl.federated_aggregator.norm_clip_aggregators import WeakDPAggregator

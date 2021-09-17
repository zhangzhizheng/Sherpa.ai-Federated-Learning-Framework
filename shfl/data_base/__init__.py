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

from shfl.data_base import data_base
from shfl.data_base.data_base import DataBase
from shfl.data_base.data_base import LabeledDatabase
from shfl.data_base.emnist import Emnist
from shfl.data_base.fashion_mnist import FashionMnist
from shfl.data_base.california_housing import CaliforniaHousing
from shfl.data_base.iris import Iris
from shfl.data_base.cifar import Cifar10
from shfl.data_base.cifar import Cifar100
from shfl.data_base.lfw import Lfw
from shfl.data_base.purchase100 import Purchase100
from shfl.data_base.federated_emnist import FederatedEmnist

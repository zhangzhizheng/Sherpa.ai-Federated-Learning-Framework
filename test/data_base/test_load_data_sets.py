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

import pytest

from shfl.data_base.california_housing import CaliforniaHousing
from shfl.data_base.cifar import Cifar100
from shfl.data_base.cifar import Cifar10
from shfl.data_base.emnist import Emnist
from shfl.data_base.federated_emnist import FederatedEmnist
from shfl.data_base.fashion_mnist import FashionMnist
from shfl.data_base.iris import Iris
from shfl.data_base.lfw import Lfw
# from shfl.data_base.purchase100 import Purchase100


@pytest.mark.parametrize("data_set",
                         [CaliforniaHousing,
                          Cifar100,
                          Cifar10,
                          Emnist,
                          FederatedEmnist,
                          FashionMnist,
                          Iris,
                          Lfw])
                          # Purchase100])
def test_data_set_load(data_set):
    """Tests the dataset loading."""
    data = data_set()
    train_data, train_labels, test_data, test_labels = data.load_data()

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)

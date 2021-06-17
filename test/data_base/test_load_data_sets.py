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
    data.load_data()
    train_data, train_labels, test_data, test_labels = data.data

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)

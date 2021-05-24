from enum import Enum
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from scipy.io import loadmat

from shfl.data_base.data_base import DataBase


class FederatedEmnist(DataBase):
    """Loads the EMNIST federated dataset.

    Implements base class [DataBase](./#database-class).

    # Arguments:
        split: String specifying the split of the original EMNIST dataset
            between 'DIGITS', 'LETTERS' and 'MNIST' (default to 'DIGITS').

    # References:
        [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
    """

    def __init__(self, split='DIGITS'):
        super().__init__()
        self._type = split

    def load_data(self):

        file_hash_ = Md5Hash[self._type].value

        path_dataset = get_file(
            'emnist-digits',
            origin='https://github.com/sherpaai/'
                   'federated-emnist-dataset/blob/master/datasets/emnist-' +
                   self._type.lower() +
                   '.mat?raw=true',
            file_hash=file_hash_,
            extract=True,
            cache_dir='~/.sherpa-ai')

        dataset = loadmat(path_dataset)['dataset']

        writers = dataset['train'][0, 0]['writers'][0, 0]
        data = np.reshape(
            dataset['train'][0, 0]['images'][0, 0],
            (-1, 28, 28, 1),
            order='F')
        self._train_data = [(writers[i][0], v) for i, v in enumerate(data)]
        self._train_labels = np.reshape(
            np.eye(10)[dataset['train'][0, 0]['labels'][0, 0]],
            (len(self._train_data), 10))

        self._test_data = np.reshape(
            dataset['test'][0, 0]['images'][0, 0],
            (-1, 28, 28, 1),
            order='F')
        self._test_labels = np.reshape(
            np.eye(10)[dataset['test'][0, 0]['labels'][0, 0]],
            (self._test_data.shape[0], 10))

        return self.data


class Md5Hash(Enum):
    """Enum Class for registering the file md 5 hash.
    """
    DIGITS = "5a18b33e88e3884e79f8b2d6274564d7"
    LETTERS = "b9eddc3e325dee05b65fb21ee45da52f"
    MNIST = "f1981b6bbe3451ba76b2078633f03b95"

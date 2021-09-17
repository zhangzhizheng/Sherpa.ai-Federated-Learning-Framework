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

import random
import numpy as np
import tensorflow as tf
import torch


class Reproducibility:
    """Ensures reproducibility.

    Singleton class.

    The server initializes this class and the clients simply get a seed.
    The reproducibility only works for executions on CPU.
    Many operations on GPU (e.g. convolutions) are not
    deterministic, and thus they don't replicate.

    # Arguments:
        seed: Integer, the main seed for server.

    # Properties:
        seed: Returns server's seed.
        seeds: Returns all seeds.

    # Example:
        ```{python}
            from shfl.private.reproducibility import Reproducibility

            # All executions will be the same:
            Reproducibility(567)
        ```
    """
    __instance = None

    @staticmethod
    def get_instance():
        """Returns the singleton class.

        # Returns:
            instance: Singleton instance class.
        """
        if Reproducibility.__instance is None:
            Reproducibility()
        return Reproducibility.__instance

    def __init__(self, seed=None):
        """Virtually private constructor.
        """
        if Reproducibility.__instance is not None:
            raise Exception("This class is a singleton")

        self.__seed = seed
        self.__seeds = {'server': self.__seed}
        Reproducibility.__instance = self

        if self.__seed is not None:
            self.set_seed('server')

    def set_seed(self, node_id):
        """Sets server's and clients' seeds.

        # Arguments:
            id: 'server' in server node and ID in clients' nodes.
        """
        if node_id not in self.__seeds.keys():
            self.__seeds[node_id] = np.random.randint(2 ** 32 - 1)
        np.random.seed(self.__seeds[node_id])
        random.seed(self.__seeds[node_id])
        tf.random.set_seed(self.__seeds[node_id])
        torch.manual_seed(self.__seeds[node_id])

    @property
    def seed(self):
        """Returns server's seed."""
        return self.__seed

    @property
    def seeds(self):
        """Returns all seeds."""
        return self.__seeds

    def delete_instance(self):
        """Removes the singleton instance.

        This method is necessary for tests and it is
        not recommended for normal use.
        """
        if Reproducibility.__instance is not None:
            del self.__seed
            del self.__seeds
            Reproducibility.__instance = None

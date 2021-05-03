from shfl.federated_aggregator.weighted_fedavg_aggregator import WeightedFedAvgAggregator

import numpy as np


class IowaFederatedAggregator(WeightedFedAvgAggregator):
    """Performs an IOWA weighted aggregation.

    It implements the class
    [WeightedFedAvgAggregator](./#weightedfedavgaggregator-class).

    # References:
        [Dynamic federated learning model for identifying
        adversarial clients](https://arxiv.org/abs/2007.15030)
    """

    def __init__(self):
        super().__init__()
        self._a = 0
        self._b = 0
        self._c = 0
        self._y_b = 0
        self._k = 0
        self._performance = None
        self._dynamic = None

    def set_ponderation(self, performance, dynamic=True,
                        a=0, b=0.2, c=0.8, y_b=0.4, k=3/4):
        """Computes the weight of each client based on the performance vector.

        # Arguments:
            performance: Vector containing the performance
                of each local client in a validation set.
            dynamic: Optional; Boolean indicating whether we use the dynamic
                or static version (default is True).
            a: Optional; First argument of linguistic quantifier (default is 0).
            b: Optional; Second argument of linguistic quantifier (default is 0.2).
            c: Optional; Third argument of linguistic quantifier (default is 0.8).
            y_b: Optional; Fourth argument of linguistic quantifier (default is 0.4).
            k: Optional; Distance param of the dynamic version (default is 3/4).
        """
        self._a = a
        self._b = b
        self._c = c
        self._y_b = y_b
        self._k = k
        self._performance = performance
        self._dynamic = dynamic

        self._percentage = self.get_ponderation_weights()

    def q_function(self, x):
        """Returns the weights for the OWA operator.

        # Arguments:
            x: Value of the ordering function u
                (ordered performance of each local model).

        # Returns:
            ponderation_weights: The weight of each client.
        """
        if x <= self._a:
            return 0
        elif x <= self._b:
            return (x - self._a) / (self._b - self._a) * self._y_b
        elif x <= self._c:
            return (x - self._b) / (self._c - self._b) * \
                   (1 - self._y_b) + self._y_b
        else:
            return 1

    def get_ponderation_weights(self):
        """Returns the linguistic quantifier (Q function) for each value x.

        # Returns:
            ponderation_weights: The weight of each client.
        """

        ordered_idx = np.argsort(-self._performance)
        self._performance = self._performance[ordered_idx]
        num_clients = len(self._performance)

        ponderation_weights = np.zeros(num_clients)

        if self._dynamic:
            max_distance = self._performance[0] - self._performance[-1]
            vector_distances = np.array(
                [self._performance[0] - self._performance[i]
                 for i in range(num_clients)])

            is_outlier = np.array(
                [vector_distances[i] > self._k * max_distance
                 for i in range(num_clients)])
            num_outliers = len(is_outlier[is_outlier is True])

            self._c = 1 - num_outliers / num_clients
            self._b = self._b * self._c

        for i in range(num_clients):
            ponderation_weights[i] = self.q_function((i + 1) / num_clients) - \
                                     self.q_function(i / num_clients)

        return ponderation_weights

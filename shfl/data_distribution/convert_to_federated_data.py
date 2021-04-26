from shfl.private.data import LabeledData
from shfl.private.federated_operation import FederatedData


def convert_to_federated_data(distributed_data, distributed_labels=None):
    """Converts distributed data to a federated_data object.

    The input should be a list where each element
    contains one single client's private data.

        # Arguments:
            distributed_data: List, each element contains
                one client's data.
            distributed_labels: Optional; List, each element contains
                one client's labels.

        # Returns:
            federated_data: Object of type federated_data.
    """
    if distributed_labels is None:
        distributed_labels = [None] * len(distributed_data)

    federated_data = FederatedData()
    for node_data, node_labels in zip(distributed_data, distributed_labels):
        node_labeled_data = LabeledData(node_data, node_labels)
        federated_data.add_data_node(node_labeled_data)

    return federated_data

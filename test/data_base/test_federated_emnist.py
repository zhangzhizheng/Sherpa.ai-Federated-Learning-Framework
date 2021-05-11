from shfl.data_base.federated_emnist import FederatedEmnist


def test_federated_emnist():
    data = FederatedEmnist('error_type')
    data.load_data()
    train_data, train_labels, test_data, test_labels = data.data

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)

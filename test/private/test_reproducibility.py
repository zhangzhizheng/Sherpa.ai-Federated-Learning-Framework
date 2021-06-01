import pytest
import numpy as np

from shfl.private import Reproducibility


@pytest.fixture(name="reproducibility_instance")
def fixture_reproducibility_instance():
    """Instantiates the reproducibility class and returns its seed."""
    Reproducibility.get_instance().delete_instance()
    seed = 1234
    Reproducibility(seed)

    return seed


def test_initialization(reproducibility_instance):
    """Checks that the reproducibility instance is correctly initialized."""
    seed = reproducibility_instance

    assert Reproducibility.get_instance().seed == seed
    assert Reproducibility.get_instance().seeds['server'] == seed
    assert np.random.get_state()[1][0] == seed


def test_reproducibility_singleton(reproducibility_instance):
    """Checks that the reproducibility class is a singleton."""
    _ = reproducibility_instance

    with pytest.raises(Exception):
        Reproducibility()


def test_set_seed(reproducibility_instance):
    """Checks that the reproducibility seed is correctly set to a specific node."""
    _ = reproducibility_instance

    node_id = 'ID0'
    Reproducibility.get_instance().set_seed(node_id)

    assert Reproducibility.get_instance().seeds[node_id]

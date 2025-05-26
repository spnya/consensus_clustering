import pytest
import numpy as np
from task_queue_api import generate_custom_synthetic_data
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_uniform_distribution():
    config = {
        "distribution": "uniform",
        "n_samples": 100,
        "n_features": 3,
        "low": -5,
        "high": 5,
        "random_state": 42
    }
    X, y = generate_custom_synthetic_data(config)
    assert X.shape == (100, 3)
    assert y.shape == (100,)
    assert (X >= -5).all() and (X <= 5).all()

def test_zipf_distribution():
    config = {
        "distribution": "zipf",
        "a": 2.0,
        "n_samples": 150,
        "n_features": 2,
        "random_state": 42
    }
    X, y = generate_custom_synthetic_data(config)
    assert X.shape == (150, 2)
    assert y.shape == (150,)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

def test_invalid_distribution():
    config = {
        "distribution": "invalid_dist",
        "n_samples": 100,
        "n_features": 2
    }
    with pytest.raises(ValueError, match="Unsupported distribution"):
        generate_custom_synthetic_data(config)

def test_zero_samples():
    config = {
        "distribution": "uniform",
        "n_samples": 0,
        "n_features": 2
    }
    X, y = generate_custom_synthetic_data(config)
    assert X.shape == (0, 2)
    assert y.shape == (0,)

"""Tests for baseline distance and brute-force index."""
import numpy as np
import pytest
from baseline.distance import euclidean_distance, cosine_distance


def test_euclidean_distance():
    assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0
    assert euclidean_distance(np.array([1, 1]), np.array([1, 1])) == 0.0


def test_cosine_distance():
    assert cosine_distance(np.array([1, 0]), np.array([2, 0])) == 0.0
    assert cosine_distance(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(2.0)
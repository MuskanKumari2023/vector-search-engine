"""Tests for baseline distance and brute-force index."""
import numpy as np
import json
import pytest
from pathlib import Path
from baseline.distance import euclidean_distance, cosine_distance
from baseline.bruteforce import BruteForceIndex
from data.ground_truth import compute_ground_truth


def test_euclidean_distance():
    assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0
    assert euclidean_distance(np.array([1, 1]), np.array([1, 1])) == 0.0


def test_cosine_distance():
    assert cosine_distance(np.array([1, 0]), np.array([2, 0])) == 0.0
    assert cosine_distance(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(2.0)

def test_bruteforce_insert_returns_sequential_ids():
    index = BruteForceIndex()
    id0 = index.insert(np.array([0.0, 0.0]))
    id1 = index.insert(np.array([1.0, 0.0]))
    id2 = index.insert(np.array([0.0, 1.0]))
    assert id0 == 0
    assert id1 == 1
    assert id2 == 2
    assert len(index.vectors) == 3

def test_bruteforce_search_hand_verifiable_five_vectors():
    """Roadmap check: correct top-k on 5 simple vectors."""
    index = BruteForceIndex()
    vectors = [
        np.array([0.0, 0.0]),   # id 0
        np.array([3.0, 4.0]),   # id 1
        np.array([1.0, 0.0]),   # id 2
        np.array([0.0, 1.0]),   # id 3
        np.array([10.0, 10.0]), # id 4
    ]
    for vector in vectors:
        index.insert(vector)
    results = index.search(np.array([0.0, 0.0]), k=3)
    # Expected distances from [0,0]:
    # id 0 -> 0.0
    # id 2 -> 1.0
    # id 3 -> 1.0
    # id 1 -> 5.0
    # id 4 -> sqrt(200) ~ 14.14
    assert results[0] == (0, 0.0)
    assert results[1][1] == pytest.approx(1.0)
    assert results[2][1] == pytest.approx(1.0)
    assert {results[1][0], results[2][0]} == {2, 3}

def test_bruteforce_search_sorted_closest_first():
    index = BruteForceIndex()
    index.insert(np.array([5.0, 0.0]))
    index.insert(np.array([1.0, 0.0]))
    index.insert(np.array([3.0, 0.0]))
    results = index.search(np.array([0.0, 0.0]), k=3)
    assert results[0][0] == 1  # distance 1.0
    assert results[1][0] == 2  # distance 3.0
    assert results[2][0] == 0  # distance 5.0
    assert results[0][1] <= results[1][1] <= results[2][1]

def test_bruteforce_search_k_larger_than_index():
    index = BruteForceIndex()
    index.insert(np.array([1.0, 2.0]))
    index.insert(np.array([3.0, 4.0]))
    results = index.search(np.array([0.0, 0.0]), k=10)
    assert len(results) == 2

def test_bruteforce_search_with_sift_vectors():
    """Integration-style check: real vectors + brute-force search runs."""
    from data.generate_vectors import load_sift_vectors, DEFAULT_SIFT_DIR
    if not (DEFAULT_SIFT_DIR / "siftsmall_base.fvecs").exists():
        pytest.skip("cached SIFT file not found")
    vectors = load_sift_vectors(100)
    index = BruteForceIndex()
    for vector in vectors:
        index.insert(vector)
    results = index.search(vectors[0], k=5)
    assert len(results) == 5
    assert results[0][0] == 0          # query equals indexed vector 0
    assert results[0][1] == pytest.approx(0.0)
    assert all(dist >= 0 for _, dist in results)

def test_compute_ground_truth_small_hand_verifiable(tmp_path: Path):
    index = BruteForceIndex()
    vectors = [
        np.array([0.0, 0.0]),
        np.array([3.0, 4.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    for vector in vectors:
        index.insert(vector)
    queries = [np.array([0.0, 0.0])]
    output_path = tmp_path / "ground_truth.json"
    result = compute_ground_truth(index, queries, k=2, output_path=str(output_path))
    assert result == {"0": [0, 2]}  # or [0, 3] if id 3 ties at dist 1.0 with k=3
    # For k=2: nearest is id 0 (dist 0), then id 2 or 3 (dist 1)
    assert result["0"][0] == 0
    assert result["0"][1] in {2, 3}
    with output_path.open() as handle:
        saved = json.load(handle)
    assert saved == result
def test_compute_ground_truth_multiple_queries(tmp_path: Path):
    index = BruteForceIndex()
    index.insert(np.array([0.0, 0.0]))
    index.insert(np.array([1.0, 0.0]))
    index.insert(np.array([10.0, 10.0]))
    queries = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    output_path = tmp_path / "ground_truth.json"
    result = compute_ground_truth(index, queries, k=1, output_path=str(output_path))
    assert result == {"0": [0], "1": [1]}
    assert output_path.exists()
@pytest.mark.skipif(
    not Path("data/cache/small/siftsmall_base.fvecs").exists(),
    reason="cached SIFT file not found",
)
def test_compute_ground_truth_with_sift(tmp_path: Path):
    from data.generate_vectors import load_sift_vectors, load_sift_queries
    vectors = load_sift_vectors(1_000)
    index = BruteForceIndex()
    for vector in vectors:
        index.insert(vector)
    queries = load_sift_queries()[:5]
    output_path = tmp_path / "ground_truth_k10.json"
    result = compute_ground_truth(index, queries, k=10, output_path=str(output_path))
    assert len(result) == 5
    for query_id in map(str, range(5)):
        assert len(result[query_id]) == 10
        assert all(isinstance(nid, int) for nid in result[query_id])
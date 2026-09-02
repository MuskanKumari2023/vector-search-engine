import numpy as np
import pytest

from baseline.bruteforce import BruteForceIndex
from eval.benchmark import benchmark_scales, measure_latency
from eval.recall import mean_recall_at_k, recall_at_k
from hnsw.hnsw import HNSWIndex


def test_recall_at_k_perfect_match():
    assert recall_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k([1, 99, 3], [1, 2, 3], k=3) == pytest.approx(2 / 3)


def test_recall_at_k_no_match():
    assert recall_at_k([10, 11, 12], [1, 2, 3], k=3) == 0.0


def test_mean_recall_at_k():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((500, 16)).astype(np.float32)
    queries = rng.standard_normal((20, 16)).astype(np.float32)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    brute = BruteForceIndex()
    for v in vectors:
        index.insert(v)
        brute.insert(v)

    ground_truth = {}
    k = 5
    for i, q in enumerate(queries):
        ground_truth[str(i)] = [nid for nid, _ in brute.search(q, k=k)]

    recall = mean_recall_at_k(index, queries, ground_truth, k=k, ef_search=100)
    assert 0.0 < recall <= 1.0


def test_measure_latency_returns_positive_ms():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((200, 16)).astype(np.float32)
    queries = rng.standard_normal((10, 16)).astype(np.float32)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    for v in vectors:
        index.insert(v)

    result = measure_latency(index, queries, k=5, n_trials=2, ef_search=50)
    assert result["avg_latency_ms"] > 0
    assert result["n_queries"] == 10


def test_benchmark_scales_recall_is_reasonable():
    rng = np.random.default_rng(5)
    queries = rng.standard_normal((20, 16)).astype(np.float32)

    results = benchmark_scales(
        sizes=[500],
        dim=16,
        queries=queries,
        k=5,
        ef_search=100,
        seed=5,
    )

    assert results[500]["recall"] > 0.5
    assert results[500]["hnsw_latency_ms"] > 0
    assert results[500]["brute_latency_ms"] > 0
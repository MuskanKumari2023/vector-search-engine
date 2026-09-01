"""Tests for NSW single-layer graph."""
import numpy as np
import pytest
from baseline.bruteforce import BruteForceIndex
from hnsw.nsw import NSWGraph


def test_insert_first_node_has_no_neighbors():
    graph = NSWGraph(M=3)
    node_id = graph.insert(np.array([0.0, 0.0]))
    assert node_id == 0
    assert graph.neighbors[0] == []


def test_insert_connects_to_m_nearest_neighbors():
    graph = NSWGraph(M=2)
    graph.insert(np.array([0.0, 0.0]))   # id 0
    graph.insert(np.array([10.0, 0.0]))  # id 1
    graph.insert(np.array([1.0, 0.0]))   # id 2 — nearest to 0 and 1

    # id 2 should connect to id 0 (dist 1) and id 1 (dist 9)
    assert set(graph.neighbors[2]) == {0, 1}
    assert 2 in graph.neighbors[0]
    assert 2 in graph.neighbors[1]


def test_insert_matches_bruteforce_m_nearest():
    """Task 2.1 done-when: edges match true M-nearest neighbors."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    M = 4
    graph = NSWGraph(M=M)
    for node_id, vector in enumerate(vectors):
        graph.insert(vector)
        if node_id == 0:
            continue
        prior = [(i, vectors[i]) for i in range(node_id)]
        true_neighbors = sorted(
            [(i, graph.distance_fn(vector, v)) for i, v in prior],
            key=lambda x: x[1],
        )[:M]
        true_ids = {i for i, _ in true_neighbors}
        assert set(graph.neighbors[node_id]) == true_ids


def test_greedy_search_finds_exact_neighbor_when_entry_is_close():
    graph = NSWGraph(M=2)
    graph.insert(np.array([0.0, 0.0]))
    graph.insert(np.array([5.0, 0.0]))
    graph.insert(np.array([1.0, 0.0]))

    result = graph.greedy_search(np.array([0.1, 0.0]), entry_point_id=1)
    assert result == 0  # [0,0] is closest to query


def test_greedy_search_local_minimum_trap():
    """Adversarial graph: greedy search misses true nearest neighbor."""
    graph = NSWGraph(M=1)
    graph.insert(np.array([0.0, 0.0]))   # id 0
    graph.insert(np.array([10.0, 0.0]))  # id 1 — true nearest to query
    graph.insert(np.array([5.0, 0.0]))   # id 2 — entry point
    query = np.array([9.5, 0.0])
    result = graph.greedy_search(query, entry_point_id=2)
    # From 2: only neighbor is 0 (dist 9.5), which is worse than current (4.5)
    # Stuck at 2 — true nearest is 1 (dist 0.5) but not reachable
    assert result == 2
    assert result != 1

@pytest.mark.skipif(
    not __import__("pathlib").Path("data/cache/small/siftsmall_base.fvecs").exists(),
    reason="cached SIFT file not found",
)
def test_greedy_search_decent_overlap_with_ground_truth():
    """Task 2.2 done-when: result often in true top-5."""
    from data.generate_vectors import load_sift_vectors, load_sift_queries

    vectors = load_sift_vectors(100)
    queries = load_sift_queries()[:20]

    graph = NSWGraph(M=8)
    brute = BruteForceIndex()
    for vector in vectors:
        graph.insert(vector)
        brute.insert(vector)

    hits_top1 = 0
    hits_top5 = 0
    for query in queries:
        result = graph.greedy_search(query, entry_point_id=0)
        top5_ids = {nid for nid, _ in brute.search(query, k=5)}
        top1_id = brute.search(query, k=1)[0][0]
        if result == top1_id:
            hits_top1 += 1
        if result in top5_ids:
            hits_top5 += 1

    assert hits_top5 >= 10  # decent overlap, not perfect
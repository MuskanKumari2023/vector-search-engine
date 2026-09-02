"""Tests for HNSW index construction."""
from collections import Counter, deque
from baseline.bruteforce import BruteForceIndex
import numpy as np
import pytest

from hnsw.hnsw import HNSWIndex

def _undirected_adjacency(index, layer: int) -> dict[int, set[int]]:
    """Adjacency where an edge counts if A->B or B->A.

    HNSW pruning is one-directional, so a layer is expected to hold
    asymmetric edges; connectivity is only meaningful undirected.
    """
    adjacency: dict[int, set[int]] = {nid: set() for nid in index.nodes}
    for node_id, node in index.nodes.items():
        for neighbor_id in node.neighbors.get(layer, []):
            adjacency[node_id].add(neighbor_id)
            adjacency[neighbor_id].add(node_id)
    return adjacency

def test_assign_layer_exponential_distribution():
    index = HNSWIndex(dim=8, M=16)
    layers = [index._assign_layer() for _ in range(2000)]
    hist = Counter(layers)
    assert hist[0] > hist.get(1, 0) > hist.get(2, 0)
    assert hist[0] > 500  # majority at layer 0


def test_insert_first_node():
    index = HNSWIndex(dim=4)
    node_id = index.insert(np.array([1.0, 2.0, 3.0, 4.0]))
    assert node_id == 0
    assert index.entry_point == 0
    assert index.max_layer >= 0


def test_layer0_connectivity():
    """Task 3.2: every node at layer 0 is reachable via BFS."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((500, 16)).astype(np.float32)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    for v in vectors:
        index.insert(v)

    # BFS at layer 0 from entry point
    adjacency = _undirected_adjacency(index, layer=0)
    start = index.entry_point
    visited = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nid in adjacency[cur]:
            if nid not in visited:
                visited.add(nid)
                queue.append(nid)

    assert len(visited) == len(index.nodes)


def test_no_self_loops_or_isolated_nodes():
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((300, 16)).astype(np.float32)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    for v in vectors:
        index.insert(v)

    adjacency = _undirected_adjacency(index, layer=0)
    for node_id, node in index.nodes.items():
        assert node_id not in node.neighbors[0]
        assert adjacency[node_id], f"node {node_id} is isolated at layer 0"


def test_neighbor_counts_respect_layer_caps():
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((300, 16)).astype(np.float32)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    for v in vectors:
        index.insert(v)

    for node in index.nodes.values():
        for layer, neighbors in node.neighbors.items():
            cap = index.M_max0 if layer == 0 else index.M
            assert len(neighbors) <= cap


def test_insert_5000_vectors_no_errors():
    """Task 3.3: construction completes for 5000+ vectors."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((5000, 32)).astype(np.float32)

    index = HNSWIndex(dim=32, M=16, ef_construction=100)
    for v in vectors:
        index.insert(v)

    assert len(index.nodes) == 5000


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/cache/small/siftsmall_base.fvecs").exists(),
    reason="cached SIFT file not found",
)
def test_insert_sift_subset():
    from data.generate_vectors import load_sift_vectors

    vectors = load_sift_vectors(1000)
    index = HNSWIndex(dim=128, M=8, ef_construction=50)
    for v in vectors:
        index.insert(v)
    assert len(index.nodes) == 1000

def _recall_at_k(retrieved_ids, ground_truth_ids, k):
    gt = set(ground_truth_ids[:k])
    retrieved = set(retrieved_ids[:k])
    return len(gt & retrieved) / k
    
def test_search_empty_index():
    index = HNSWIndex(dim=4)
    assert index.search(np.array([1.0, 2.0, 3.0, 4.0]), k=5) == []

def test_search_returns_top_k():
    """Task 4.1: search returns top-k on a built index."""
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((200, 16)).astype(np.float32)
    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    for vector in vectors:
        index.insert(vector)
    query = vectors[10]
    results = index.search(query, k=10, ef_search=50)
    assert len(results) == 10
    assert results[0][0] == 10          # exact match
    assert results[0][1] == pytest.approx(0.0)
    assert results[0][1] <= results[-1][1]  # sorted closest first

def test_search_on_large_index():
    """Task 4.1: 5000-vector index, no errors."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((5000, 32)).astype(np.float32)
    index = HNSWIndex(dim=32, M=16, ef_construction=100)
    for vector in vectors:
        index.insert(vector)
    results = index.search(vectors[0], k=10, ef_search=50)
    assert len(results) == 10
    assert results[0][1] == pytest.approx(0.0)

def test_search_recall_improves_with_ef_search():
    """Task 4.2: higher ef_search → higher recall vs brute force."""
    rng = np.random.default_rng(21)
    vectors = rng.standard_normal((1000, 32)).astype(np.float32)
    queries = rng.standard_normal((30, 32)).astype(np.float32)
    index = HNSWIndex(dim=32, M=16, ef_construction=100)
    brute = BruteForceIndex()
    for vector in vectors:
        index.insert(vector)
        brute.insert(vector)
    k = 10
    recalls = {}
    for ef_search in (10, 50, 100, 200):
        total = 0.0
        for query in queries:
            retrieved = [nid for nid, _ in index.search(query, k=k, ef_search=ef_search)]
            ground_truth = [nid for nid, _ in brute.search(query, k=k)]
            total += _recall_at_k(retrieved, ground_truth, k)
        recalls[ef_search] = total / len(queries)
    assert recalls[10] <= recalls[50] <= recalls[100] <= recalls[200]
    assert recalls[200] > recalls[10]
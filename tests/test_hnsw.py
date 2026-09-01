"""Tests for HNSW index construction."""
from collections import Counter, deque

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
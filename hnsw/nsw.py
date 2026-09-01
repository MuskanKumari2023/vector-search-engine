import numpy as np
from baseline.distance import euclidean_distance

class NSWGraph:
    """Single-layer navigable small world graph — the stepping stone
    before full HNSW. No layers, no randomness yet."""

    def __init__(self, M: int = 8, distance_fn=None):
        self.M = M
        self.distance_fn = distance_fn or euclidean_distance
        self.vectors: dict[int, np.ndarray] = {}
        self.neighbors: dict[int, list[int]] = {}
        self.next_id = 0

    def insert(self, vector: np.ndarray) -> int:
        """
        Task 2.1. Connect the new vector to its M nearest ALREADY-INSERTED
        neighbors (brute-force distance is fine here — correctness over
        speed at this stage). Edges must be bidirectional: if new node N
        connects to existing node X, X also gets an edge back to N.

        Order dependency is expected and fine — the first few insertions
        will have fewer than M neighbors available, that's not a bug.
        """
        node_id = self.next_id
        self.next_id += 1
        vector = np.asarray(vector, dtype=float)
        self.vectors[node_id] = vector
        self.neighbors[node_id] = []
        if node_id == 0:
            return node_id  # first node — no existing neighbors
        # Find M nearest among already-inserted nodes (exclude self)
        candidates = []
        for existing_id, existing_vec in self.vectors.items():
            if existing_id == node_id:
                continue
            dist = self.distance_fn(vector, existing_vec)
            candidates.append((existing_id, dist))
        candidates.sort(key=lambda x: x[1])
        nearest = [nid for nid, _ in candidates[: self.M]]
        # Bidirectional edges
        for neighbor_id in nearest:
            self.neighbors[node_id].append(neighbor_id)
            self.neighbors[neighbor_id].append(node_id)
        return node_id

    def greedy_search(self, query: np.ndarray, entry_point_id: int) -> int:
        """
        Task 2.2. From entry_point_id, repeatedly move to whichever
        neighbor is closest to the query, until no neighbor improves on
        the current position. Returns the final node's ID.

        This WILL sometimes fail to find the true nearest neighbor (local
        minimum trap) — that's expected, not a bug. Don't try to "fix"
        this in Phase 2; it's what the hierarchy in Phase 3 addresses.
        """
        query = np.asarray(query, dtype=float)
        current_id = entry_point_id
        current_dist = self.distance_fn(query, self.vectors[current_id])
        while True:
            improved = False
            for neighbor_id in self.neighbors[current_id]:
                neighbor_dist = self.distance_fn(query, self.vectors[neighbor_id])
                if neighbor_dist < current_dist:
                    current_id = neighbor_id
                    current_dist = neighbor_dist
                    improved = True
                    break  # greedy: take first improving neighbor
            if not improved:
                break
        return current_id

from baseline.distance import euclidean_distance
import numpy as np

class BruteForceIndex:
    """Your correctness oracle for the rest of the project."""

    def __init__(self, distance_fn=euclidean_distance):
        self.vectors = []
        self.distance_fn = distance_fn

    def insert(self, vector) -> int:
        """Task 1.2. Return the assigned integer ID."""
        node_id = len(self.vectors)
        self.vectors.append(np.asarray(vector, dtype=float))
        return node_id

    def search(self, query, k: int) -> list[tuple[int, float]]:
        """Task 1.2. Return top-k (id, distance), sorted closest first.
        O(n) — that's expected, it's the whole point of comparison."""
        query = np.asarray(query, dtype=float)
        distances = [
            (i, self.distance_fn(query, vec))
            for i, vec in enumerate(self.vectors)
        ]
        distances.sort(key=lambda x: x[1])  # closest first
        return distances[:k]

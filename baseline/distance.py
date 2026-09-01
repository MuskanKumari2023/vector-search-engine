import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance. Task 0.2."""
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity — a distance, lower is closer."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0  # both zero vectors: identical, distance 0
    similarity = np.dot(a, b) / denom
    return float(1.0 - similarity)
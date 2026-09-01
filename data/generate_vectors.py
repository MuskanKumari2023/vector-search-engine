import struct
import numpy as np
from pathlib import Path

def read_fvecs(path, max_vectors=None):
    vectors = []
    with open(path, "rb") as f:
        while True:
            if max_vectors and len(vectors) >= max_vectors:
                break
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            vectors.append(vec)
    return np.stack(vectors)  # shape (N, 128)

DEFAULT_SIFT_DIR = Path(__file__).parent / "cache" / "small"

def load_sift_vectors(n=10_000, cache_dir=None):
    cache_dir = Path(cache_dir or DEFAULT_SIFT_DIR)
    path = cache_dir / "siftsmall_base.fvecs"
    return read_fvecs(path, max_vectors=n)

def load_sift_queries(cache_dir=None):
    cache_dir = Path(cache_dir or DEFAULT_SIFT_DIR)
    path = cache_dir / "siftsmall_query.fvecs"
    return read_fvecs(path)  # shape (100, 128)
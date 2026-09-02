import shutil
import struct
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

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
SIFTSMALL_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
SIFTSMALL_FILES = ("siftsmall_base.fvecs", "siftsmall_query.fvecs")

def download_siftsmall(cache_dir=None) -> Path:
    """Fetch the 10k-vector SIFT corpus (~5 MB) unless it is already cached.

    INRIA serves this over FTP only — the corpus-texmex.irisa.fr HTTP path
    404s. If your network blocks FTP, fetch siftsmall.tar.gz by hand and drop
    the two .fvecs files into cache_dir; this function then no-ops.
    """
    cache_dir = Path(cache_dir or DEFAULT_SIFT_DIR)
    if all((cache_dir / name).exists() for name in SIFTSMALL_FILES):
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SIFT10K to {cache_dir} (~5 MB, one time only)...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "siftsmall.tar.gz"
        urllib.request.urlretrieve(SIFTSMALL_URL, archive)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                # Copy the payload out by hand rather than extracting: the
                # archive nests files under siftsmall/, and matching on the
                # basename keeps any archive path from reaching the filesystem.
                name = Path(member.name).name
                if member.isfile() and name in SIFTSMALL_FILES:
                    with tar.extractfile(member) as src:
                        with open(cache_dir / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)

    missing = [n for n in SIFTSMALL_FILES if not (cache_dir / n).exists()]
    if missing:
        raise RuntimeError(
            f"SIFT download completed but {missing} are missing. Fetch "
            f"{SIFTSMALL_URL} manually and extract into {cache_dir}."
        )
    return cache_dir

def load_sift_vectors(n=10_000, cache_dir=None):
    cache_dir = download_siftsmall(cache_dir)
    path = cache_dir / "siftsmall_base.fvecs"
    return read_fvecs(path, max_vectors=n)

def load_sift_queries(cache_dir=None):
    cache_dir = download_siftsmall(cache_dir)
    path = cache_dir / "siftsmall_query.fvecs"
    return read_fvecs(path)  # shape (100, 128)
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from baseline.bruteforce import BruteForceIndex
from eval.recall import mean_recall_at_k
from hnsw.hnsw import HNSWIndex

def _run_search(index, query, k: int, ef_search: int = 50):
    if isinstance(index, HNSWIndex):
        return index.search(query, k=k, ef_search=ef_search)
    return index.search(query, k=k)

def measure_latency(index, queries: list, k: int, n_trials: int = 3,warmup: int = 3,ef_search: int = 50) -> dict:
    """
    Time search across all queries, averaged over n_trials.
    Returns per-query average latency in milliseconds.
    """
    query_list = [np.asarray(q, dtype=float) for q in queries]
    # Warm-up (cold cache not representative)
    for q in query_list[:warmup]:
        _run_search(index, q, k=k, ef_search=ef_search)
    total_seconds = 0.0
    total_queries = 0
    for _ in range(n_trials):
        start = time.perf_counter()
        for q in query_list:
            _run_search(index, q, k=k, ef_search=ef_search)
        total_seconds += time.perf_counter() - start
        total_queries += len(query_list)
    avg_ms_per_query = (total_seconds / total_queries) * 1000
    return {
        "avg_latency_ms": avg_ms_per_query,
        "n_queries": len(query_list),
        "n_trials": n_trials,
        "k": k,
        "ef_search": ef_search,
    }

def benchmark_scales(
    sizes: list[int],
    dim: int,
    queries: list,
    k: int = 10,
    ef_search: int = 50,
    seed: int = 42,
) -> dict:
    """
    Build HNSW + brute-force at each size, measure recall and latency.
    Returns {size: {"hnsw_latency_ms": ..., "brute_latency_ms": ..., "recall": ...}}

    Ground truth is computed from brute force on the same vectors used to
    build each index — do not precompute it separately with a different RNG
    stream or recall will be meaningless.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for n in sizes:
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        hnsw = HNSWIndex(dim=dim, M=16, ef_construction=100)
        brute = BruteForceIndex()
        for v in vectors:
            hnsw.insert(v)
            brute.insert(v)
        gt = {
            str(i): [nid for nid, _ in brute.search(q, k=k)]
            for i, q in enumerate(queries)
        }
        recall = mean_recall_at_k(hnsw, queries, gt, k=k, ef_search=ef_search)
        hnsw_lat = measure_latency(hnsw, queries, k=k, ef_search=ef_search)
        brute_lat = measure_latency(brute, queries, k=k)
        results[n] = {
            "recall": recall,
            "hnsw_latency_ms": hnsw_lat["avg_latency_ms"],
            "brute_latency_ms": brute_lat["avg_latency_ms"],
        }
    return results

def plot_recall_vs_latency(results: dict, output_path: str) -> None:
    """
    Plot latency vs dataset size for HNSW vs brute force.
    results: {size: {"hnsw_latency_ms": float, "brute_latency_ms": float, "recall": float}}
    """
    sizes = sorted(results.keys())
    hnsw_lats = [results[s]["hnsw_latency_ms"] for s in sizes]
    brute_lats = [results[s]["brute_latency_ms"] for s in sizes]
    recalls = [results[s]["recall"] for s in sizes]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sizes, hnsw_lats, "o-", label="HNSW latency (ms)", color="tab:blue")
    ax1.plot(sizes, brute_lats, "s-", label="Brute-force latency (ms)", color="tab:red")
    ax1.set_xlabel("Dataset size (n)")
    ax1.set_ylabel("Avg latency per query (ms)")
    ax1.set_xscale("log")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(sizes, recalls, "^--", label="HNSW Recall@k", color="tab:green")
    ax2.set_ylabel("Recall@k")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right")
    plt.title("HNSW: Recall/latency tradeoff vs dataset size")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
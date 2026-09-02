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

def plot_recall_vs_latency(results: dict, output_path: str, k: int = 10) -> None:
    """
    Plot latency vs dataset size for HNSW vs brute force.
    results: {size: {"hnsw_latency_ms": float, "brute_latency_ms": float, "recall": float}}

    Both axes are log-scaled so the growth rates are readable as slopes:
    brute force sits near slope 1, HNSW well below it.
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
    ax1.set_yscale("log")
    ax1.legend(loc="center left")
    ax1.grid(True, alpha=0.3, which="both")
    ax2 = ax1.twinx()
    ax2.plot(sizes, recalls, "^--", label=f"HNSW Recall@{k}", color="tab:green")
    ax2.set_ylabel(f"Recall@{k}")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    plt.title("Query latency vs dataset size (log-log): HNSW grows sublinearly, brute force linearly")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def sweep_ef_search(
    index,
    queries: list,
    ground_truth: dict[str, list[int]],
    k: int = 10,
    ef_values: tuple[int, ...] = (10, 20, 50, 100, 200),
) -> dict:
    """
    Recall and latency at each ef_search on one already-built index.

    ef_search is the cheap knob: it is swept on a single graph, so the whole
    curve costs seconds rather than a rebuild per point.
    Returns {ef_search: {"recall": float, "latency_ms": float}}
    """
    results = {}
    for ef in ef_values:
        recall = mean_recall_at_k(index, queries, ground_truth, k=k, ef_search=ef)
        latency = measure_latency(index, queries, k=k, ef_search=ef)
        results[ef] = {"recall": recall, "latency_ms": latency["avg_latency_ms"]}
    return results

def plot_ef_search_tradeoff(
    sweep: dict,
    output_path: str,
    baseline_latency_ms: float | None = None,
    k: int = 10,
) -> None:
    """
    The recall/latency curve: one point per ef_search value on a single graph.

    baseline_latency_ms places brute force at recall 1.0 for scale — without
    it the reader cannot see what the approximation is buying.
    """
    ef_values = sorted(sweep.keys())
    latencies = [sweep[ef]["latency_ms"] for ef in ef_values]
    recalls = [sweep[ef]["recall"] for ef in ef_values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(latencies, recalls, "o-", color="tab:blue", label="HNSW")
    for i, (ef, x, y) in enumerate(zip(ef_values, latencies, recalls)):
        # Alternate above/below: the high-ef points sit at nearly identical
        # recall and would otherwise overprint each other.
        ax.annotate(
            f"ef_search={ef}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8) if i % 2 else (8, -16),
            fontsize=9,
        )
    if baseline_latency_ms is not None:
        ax.plot(
            [baseline_latency_ms], [1.0], "s", color="tab:red",
            markersize=9, label="Brute force (exact)",
        )
        ax.set_xlim(min(latencies) / 1.7, baseline_latency_ms * 2.2)
    ax.set_xscale("log")
    ax.set_xlabel("Avg latency per query (ms, log scale)")
    ax.set_ylabel(f"Recall@{k}")
    # Zoom to where the curve actually lives; anchoring at 0 turns the whole
    # tradeoff into a flat line pinned to the top of the axes.
    ax.set_ylim(max(0.0, min(recalls) - 0.06), 1.02)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")
    plt.title(f"HNSW recall/latency tradeoff — SIFT10K, sweeping ef_search (k={k})")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
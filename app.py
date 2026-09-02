"""Benchmark entry point: reproduces every number and chart in the README."""

import random

import numpy as np

from baseline.bruteforce import BruteForceIndex
from data.generate_vectors import load_sift_queries, load_sift_vectors
from data.ground_truth import compute_ground_truth
from eval.benchmark import (
    benchmark_scales,
    measure_latency,
    plot_ef_search_tradeoff,
    plot_recall_vs_latency,
    sweep_ef_search,
)
from eval.recall import mean_recall_at_k
from hnsw.hnsw import HNSWIndex

SEED = 42
K = 10
M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 50

# HNSW layer assignment draws from the global RNG, so an unseeded build gives
# a different graph — and a materially different recall — on every run.
random.seed(SEED)

# --- SIFT10K benchmark ---
vectors = load_sift_vectors(10_000)
queries = load_sift_queries()[:50]  # hold-out queries, not indexed vectors

brute = BruteForceIndex()
for v in vectors:
    brute.insert(v)

gt_path = "data/cache/small/ground_truth_k10.json"
ground_truth = compute_ground_truth(brute, queries, k=K, output_path=gt_path)

index = HNSWIndex(dim=vectors.shape[1], M=M, ef_construction=EF_CONSTRUCTION)
for v in vectors:
    index.insert(v)

recall = mean_recall_at_k(index, queries, ground_truth, k=K, ef_search=EF_SEARCH)
print(f"--- SIFT10K (n={len(vectors):,}, dim={vectors.shape[1]}, {len(queries)} queries) ---")
print(f"M={M}, ef_construction={EF_CONSTRUCTION}, ef_search={EF_SEARCH}")
print(f"Recall@{K}: {recall:.1%}")

# --- Recall/latency curve: sweep ef_search on the graph just built ---
brute_latency_ms = measure_latency(brute, queries, k=K)["avg_latency_ms"]
sweep = sweep_ef_search(index, queries, ground_truth, k=K)
plot_ef_search_tradeoff(sweep, "docs/ef_search_tradeoff.png", brute_latency_ms, k=K)

print(f"\n--- Recall/latency tradeoff (brute force: {brute_latency_ms:.2f} ms/query) ---")
print(f"{'ef_search':>10} {'recall@' + str(K):>10} {'latency':>10} {'speedup':>9}")
for ef, r in sorted(sweep.items()):
    speedup = brute_latency_ms / r["latency_ms"]
    print(f"{ef:>10} {r['recall']:>9.1%} {r['latency_ms']:>8.2f}ms {speedup:>8.1f}x")

# --- Scaling: how latency grows with dataset size ---
sizes = [1_000, 3_000, 10_000]
dim = 32
rng = np.random.default_rng(SEED)
synth_queries = rng.standard_normal((30, dim)).astype(np.float32)

results = benchmark_scales(sizes, dim, synth_queries, k=K, ef_search=EF_SEARCH)
plot_recall_vs_latency(results, "docs/scaling_chart.png", k=K)

print(f"\n--- Scaling on synthetic data (dim={dim}, {len(synth_queries)} queries) ---")
print(f"{'n':>8} {'recall@' + str(K):>10} {'HNSW':>10} {'brute':>10}")
for n, r in sorted(results.items()):
    print(
        f"{n:>8,} {r['recall']:>9.1%} "
        f"{r['hnsw_latency_ms']:>8.2f}ms {r['brute_latency_ms']:>8.2f}ms"
    )

print("\ncharts: docs/ef_search_tradeoff.png, docs/scaling_chart.png")

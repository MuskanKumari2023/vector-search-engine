import numpy as np

from baseline.bruteforce import BruteForceIndex
from data.generate_vectors import load_sift_queries, load_sift_vectors
from data.ground_truth import compute_ground_truth
from eval.benchmark import benchmark_scales, plot_recall_vs_latency
from eval.recall import mean_recall_at_k
from hnsw.hnsw import HNSWIndex

# --- SIFT 10k benchmark ---
vectors = load_sift_vectors(10_000)
queries = load_sift_queries()[:50]  # hold-out queries, not indexed vectors
k = 10

brute = BruteForceIndex()
for v in vectors:
    brute.insert(v)

gt_path = "data/cache/small/ground_truth_k10.json"
ground_truth = compute_ground_truth(brute, queries, k=k, output_path=gt_path)

index = HNSWIndex(dim=128, M=16, ef_construction=100)
for v in vectors:
    index.insert(v)

recall = mean_recall_at_k(index, queries, ground_truth, k=k, ef_search=50)
print(f"n={len(vectors)}: recall@{k}={recall:.1%}, ef_search=50")

# --- Synthetic scale benchmark (1k, 10k)
sizes = [1_000, 10_000]
dim = 32
rng = np.random.default_rng(42)
synth_queries = rng.standard_normal((30, dim)).astype(np.float32)

results = benchmark_scales(sizes, dim, synth_queries, k=k, ef_search=50)
plot_recall_vs_latency(results, "data/cache/benchmark_chart.png")

for n, r in results.items():
    print(f"n={n}: recall@{k}={r['recall']:.1%}, HNSW={r['hnsw_latency_ms']:.2f}ms, brute={r['brute_latency_ms']:.2f}ms")
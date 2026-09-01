import json
from pathlib import Path
from baseline.bruteforce import BruteForceIndex


def compute_ground_truth(
    index: BruteForceIndex,
    queries: list,
    k: int,
    output_path: str,
) -> None:
    """
    For each query, get brute-force top-k, save {query_id: [neighbor_ids]}
    to output_path. This file is your recall baseline for every later
    phase — Task 1.3.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(index.vectors) == 0:
        raise ValueError("index is empty")
    ground_truth: dict[str, list[int]] = {}
    for query_id, query in enumerate(queries):
        results = index.search(query, k)
        neighbor_ids = [node_id for node_id, _ in results]
        ground_truth[str(query_id)] = neighbor_ids
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(ground_truth, handle, indent=2)
    return ground_truth

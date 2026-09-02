def recall_at_k(
    retrieved_ids: list[int], ground_truth_ids: list[int], k: int
) -> float:
    """|retrieved ∩ ground_truth| / k.|"""
    if k < 1:
        raise ValueError("k must be >= 1")
    gt = set(ground_truth_ids[:k])
    retrieved = set(retrieved_ids[:k])
    return len(gt & retrieved) / k

def mean_recall_at_k(
    index,
    queries: list,
    ground_truth: dict[str, list[int]],
    k: int,
    ef_search: int = 50,
) -> float:
    """Average Recall@k across all queries."""
    total = 0.0
    for query_id, query in enumerate(queries):
        retrieved = [nid for nid, _ in index.search(query, k=k, ef_search=ef_search)]
        gt_ids = ground_truth[str(query_id)]
        total += recall_at_k(retrieved, gt_ids, k)
    return total / len(queries)

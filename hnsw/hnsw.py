import heapq
import math
import random
import numpy as np
from baseline.distance import euclidean_distance


class HNSWNode:
    def __init__(self, node_id: int, vector: np.ndarray, layer: int):
        self.id = node_id
        self.vector = vector
        self.layer = layer  # MAX layer this node exists at
        self.neighbors: dict[int, list[int]] = {
            l: [] for l in range(layer + 1)
        }


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        distance_fn=None,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # layer 0 gets a larger cap
        self.ef_construction = ef_construction
        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: int | None = None
        self.max_layer = -1
        self.distance_fn = distance_fn or euclidean_distance
        self.next_id = 0 # for node IDs
        self._m_l = 1 / math.log(M)

    def _assign_layer(self) -> int:
        """
        floor(-ln(uniform(0,1)) * self._m_l). -ln(U) for U~uniform(0,1)
        is exponentially distributed — heavily weighted near zero, long
        thin tail. That's what gives you "most nodes at layer 0,
        exponentially fewer higher up."
        """
        u = random.random()
        while u == 0.0:
            u = random.random()
        return int(math.floor(-math.log(u) * self._m_l))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
        exclude_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Algorithm 2. Greedy search at a single layer maintaining a
        candidate set of size ef. Maintain: a visited set, a min-heap of
        candidates to explore, a max-heap of the best ef results found.
        Pop the closest unvisited candidate, examine ITS neighbors at
        this layer, add unvisited ones to candidates. Stop when the
        closest remaining candidate is farther than the current worst
        result in your size-ef result set.

        exclude_id keeps the node currently being inserted out of its own
        result set — otherwise it comes back at distance 0 and burns a
        neighbor slot on a self-loop.
        """
        query = np.asarray(query, dtype=float)
        visited: set[int] = set()
        # min-heap: (distance, node_id) — explore closest first
        candidates: list[tuple[float, int]] = []
        # max-heap via negated dist: (-distance, node_id) — track best ef results
        results: list[tuple[float, int]] = []
        for ep in entry_points:
            if ep is None or ep == exclude_id:
                continue
            visited.add(ep)
            d = self._dist(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
        while candidates:
            cand_dist, cand_id = heapq.heappop(candidates)
            # Stop: closest candidate is worse than worst in result set
            if len(results) >= ef:
                worst_dist = -results[0][0]
                if cand_dist > worst_dist:
                    break
            for neighbor_id in self._layer_neighbors(cand_id, layer):
                if neighbor_id in visited or neighbor_id == exclude_id:
                    continue
                visited.add(neighbor_id)
                ndist = self._dist(query, neighbor_id)
                if len(results) < ef:
                    heapq.heappush(candidates, (ndist, neighbor_id))
                    heapq.heappush(results, (-ndist, neighbor_id))
                else:
                    worst_dist = -results[0][0]
                    if ndist < worst_dist:
                        heapq.heappush(candidates, (ndist, neighbor_id))
                        heapq.heappush(results, (-ndist, neighbor_id))
                        heapq.heappop(results)  # drop worst
        return sorted(
            [(node_id, -neg_dist) for neg_dist, node_id in results],
            key=lambda x: x[1],
        )

    def _select_neighbors_simple(
        self, candidates: list[tuple[int, float]], M: int
    ) -> list[int]:
        """Algorithm 3 — just the M closest. Task 3.3's deliberate
        simplification; note it as such in your README, don't hide it."""
        candidates = sorted(candidates, key=lambda x: x[1])
        return [node_id for node_id, _ in candidates[:M]]

    def insert(self, vector: np.ndarray) -> int:
        """
        Algorithm 1.
        1. Assign a random max layer (Task 3.1's function).
        2. Empty index → this node becomes entry_point, done.
        3. Descent phase: from entry_point at self.max_layer, greedily
           descend (ef=1) layer by layer down to this node's assigned
           layer + 1. No edges created here — the new node doesn't exist
           at these layers.
        4. Connection phase: from the node's assigned layer down through
           layer 0 — at each layer, run _search_layer with
           ef=self.ef_construction, select M (or M_max0 at layer 0)
           neighbors, connect bidirectionally.
        5. If any existing neighbor now exceeds its layer's cap, prune it
           back down via neighbor selection again.
        6. If this node's layer exceeds the old max_layer, it becomes the
           new entry_point.
        """
        vector = np.asarray(vector, dtype=float)
        node_id = self.next_id
        self.next_id += 1
        layer = self._assign_layer()
        node = HNSWNode(node_id, vector, layer)
        self.nodes[node_id] = node
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = layer
            return node_id
        entry_points = [self.entry_point]

        # 3. Descent phase: ef=1, from max_layer down to layer+1
        for lc in range(self.max_layer, layer, -1):
            found = self._search_layer(
                vector, entry_points, ef=1, layer=lc, exclude_id=node_id
            )
            if found:
                entry_points = [found[0][0]]

        # 4. Connection phase: from assigned layer down to 0
        for lc in range(min(layer, self.max_layer), -1, -1):
            cap = self._max_neighbors(lc)
            candidates = self._search_layer(
                vector,
                entry_points,
                ef=self.ef_construction,
                layer=lc,
                exclude_id=node_id,
            )
            for neighbor_id in self._select_neighbors_simple(candidates, cap):
                self._connect(node_id, neighbor_id, lc)

            # 5. Prune any neighbor that now exceeds this layer's cap
            for neighbor_id in list(self.nodes[node_id].neighbors[lc]):
                self._prune_neighbors(neighbor_id, lc)
            self._prune_neighbors(node_id, lc)

            # Carry the layer's results down as entry points — starting the
            # next layer from node_id would search from a node with no edges
            # there yet, stranding it in its own component.
            if candidates:
                entry_points = [nid for nid, _ in candidates]

        # 6. Update entry point if new node is taller
        if layer > self.max_layer:
            self.max_layer = layer
            self.entry_point = node_id
        return node_id

    def search(
        self, query: np.ndarray, k: int, ef_search: int = 50
    ) -> list[tuple[int, float]]:
        """
        Algorithm 5.
        1. Descend from entry_point at self.max_layer down to layer 1,
           ef=1 each hop — identical pattern to insertion's descent phase.
        2. At layer 0 only, run _search_layer with ef=ef_search.
        3. Return the top k results.

        Note ef_search is independent from ef_construction — construction
        cost is paid once per node; search cost is paid on every query, so
        this is the knob you'll actually tune live in Phase 5.
        """
        # TODO

    def _dist(self, query: np.ndarray, node_id: int) -> float:
        return self.distance_fn(query, self.nodes[node_id].vector)

    def _layer_neighbors(self, node_id: int, layer: int) -> list[int]:
        return self.nodes[node_id].neighbors.get(layer, [])

    def _max_neighbors(self, layer: int) -> int:
        return self.M_max0 if layer == 0 else self.M

    def _connect(self, node_id: int, neighbor_id: int, layer: int) -> None:
        if node_id == neighbor_id:
            return
        if self.nodes[node_id].layer < layer or self.nodes[neighbor_id].layer < layer:
            return
        if neighbor_id not in self.nodes[node_id].neighbors[layer]:
            self.nodes[node_id].neighbors[layer].append(neighbor_id)
        if node_id not in self.nodes[neighbor_id].neighbors[layer]:
            self.nodes[neighbor_id].neighbors[layer].append(node_id)

    def _prune_neighbors(self, node_id: int, layer: int) -> None:
        """Trim neighbor list back to layer cap using simple M-closest selection.

        Pruning is one-directional on purpose: the pruned node drops the edge
        but the other side keeps it, which is what stops a freshly inserted
        node from having every connection stripped back off it.
        """
        cap = self._max_neighbors(layer)
        node = self.nodes[node_id]
        if layer not in node.neighbors or len(node.neighbors[layer]) <= cap:
            return
        candidates = [
            (nid, self._dist(node.vector, nid)) for nid in node.neighbors[layer]
        ]
        keep = set(self._select_neighbors_simple(candidates, cap))
        node.neighbors[layer] = [nid for nid in node.neighbors[layer] if nid in keep]


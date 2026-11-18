"""Directed road graph of one trip, with cached shortest-path distances (metres).

Nodes = road-network nodes, edges = arcs (from_node -> to_node), weight = arc length.
Only arcs inside the cropped area around the track are used. Distances are
computed with Dijkstra (scipy) from a source node up to a distance limit and
cached, because the Viterbi decoder asks for many short paths.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

BASE_LIMIT_M = 600.0


class RoadGraph:
    def __init__(self, trip):
        self.trip = trip
        arcs = trip.active_arcs
        e = pd.DataFrame({"u": trip.arcs[arcs, 0], "v": trip.arcs[arcs, 1],
                          "w": np.maximum(trip.arc_len[arcs], 0.01)})
        e = e.groupby(["u", "v"], as_index=False).w.min()  # parallel arcs: keep the shortest
        n = trip.node_xy.shape[0]
        self.n = n
        self.csr = csr_matrix((e.w.to_numpy(), (e.u.to_numpy(), e.v.to_numpy())), shape=(n, n))
        self._cache = {}   # node -> (limit, {target: distance})

    def distances_from(self, sources, limit=BASE_LIMIT_M):
        """Make sure Dijkstra results from each source (up to `limit`) are cached."""
        todo = [s for s in set(int(s) for s in sources)
                if s not in self._cache or self._cache[s][0] < limit]
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            d = dijkstra(self.csr, directed=True, indices=batch, limit=limit)
            for row, s in zip(d, batch):
                idx = np.flatnonzero(np.isfinite(row))
                self._cache[s] = (limit, dict(zip(idx.tolist(), row[idx].tolist())))

    def dist(self, u, v):
        """Shortest network distance u -> v (inf if not reachable within the cached limit)."""
        if u == v:
            return 0.0
        return self._cache[int(u)][1].get(int(v), np.inf)

    def connected(self, arc_a, arc_b):
        """True if arc_b starts where arc_a ends (or it is the same arc)."""
        arcs = self.trip.arcs
        return arc_a == arc_b or arcs[arc_a, 1] == arcs[arc_b, 0]

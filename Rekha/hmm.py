"""Hidden Markov Model + Viterbi decoding for map matching.

Hidden state at GPS point i : one of the candidate arcs of point i
                              (state k of point i <-> row (point_idx=i, arc_id) of the candidate table).
Emission  log P(obs_i | arc)   classifier mode: w * log( p_arc / sum_candidates p )
                               distance mode  : -0.5 * (dist / sigma)^2
Transition log P(arc_b | arc_a) = -| d_route(a -> b) - d_gps(i-1 -> i) | / beta_eff
    d_route = network distance from the projection of point i-1 on arc a to the
              projection of point i on arc b (same arc: s_b - s_a; otherwise the rest
              of arc a + shortest path end(a) -> start(b) + s_b)
    beta_eff = beta * sqrt(max(dt, 1)): longer time gaps tolerate more mismatch.
    Moving backwards along the same arc is usually GPS jitter (e.g. a stopped car):
    up to BACKWARD_TOL_M it costs |s_b - s_a|, beyond that the distance counts double.
    (A hard ban on backward movement was tried first; on held-out training trips it lowered
    pointwise accuracy by about 6 points, because jitter forced wrong arc switches.)
    Unreachable pairs (no path within the search limit) and very poor transitions are
    floored at `max_penalty`, so the decoder never gets stuck: a disconnected jump is
    allowed but expensive.
Points without candidates have no state: they are skipped by the decoder and get
the previous decoded arc (forward fill), flagged in `decoded`.

Viterbi recurrence (log space):
    delta_i(b) = emission_i(b) + max_a [ delta_{i-1}(a) + trans(a, b) ]
    back_i(b)  = argmax_a [ ... ]
"""
import numpy as np

BACKWARD_TOL_M = 15.0     # backward movement up to this is treated as GPS jitter
BACKWARD_FACTOR = 2.0     # beyond it, backward distance counts double (soft U-turn penalty)


def emission_logp(cands, mode, proba=None, sigma=10.0, weight=1.0):
    if mode == "distance":
        return -0.5 * (cands["dist"].to_numpy() / sigma) ** 2
    if mode == "classifier":
        p = np.clip(proba, 1e-6, 1.0)
        s = cands.assign(_p=p).groupby("point_idx")["_p"].transform("sum").to_numpy()
        return weight * np.log(p / s)
    raise ValueError(mode)


def viterbi_decode(trip, graph, cands, emis, beta=5.0, max_penalty=50.0):
    """Return (pred_arc per GPS point, decoded mask)."""
    n = trip.n_points
    pred = np.full(n, -1, dtype=np.int64)
    decoded = np.zeros(n, dtype=bool)
    if len(cands) == 0:
        return pred, decoded

    order = np.argsort(cands["point_idx"].to_numpy(), kind="stable")
    pidx = cands["point_idx"].to_numpy()[order]
    arc = cands["arc_id"].to_numpy()[order]
    s_along = (cands["frac"].to_numpy() * trip.arc_len[cands["arc_id"].to_numpy()])[order]
    em = np.asarray(emis)[order]
    pts, start = np.unique(pidx, return_index=True)
    end = np.r_[start[1:], len(pidx)]

    x, y, t = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    from_node, to_node, alen = trip.arcs[:, 0], trip.arcs[:, 1], trip.arc_len

    # pre-compute shortest paths from the end node of every candidate arc
    gaps = np.hypot(np.diff(x[pts]), np.diff(y[pts]))
    graph.distances_from(to_node[np.unique(arc)])
    far = np.flatnonzero(gaps > 300)
    for j in far:  # long jumps need a larger search radius
        a_prev = arc[start[j]:end[j]]
        graph.distances_from(to_node[a_prev], limit=3 * gaps[j] + 500)

    delta = em[start[0]:end[0]].copy()
    backs = []
    for j in range(1, len(pts)):
        pa = slice(start[j - 1], end[j - 1])
        pb = slice(start[j], end[j])
        A, B = arc[pa], arc[pb]
        sA, sB = s_along[pa], s_along[pb]
        d_gps = gaps[j - 1]
        dt = max(t[pts[j]] - t[pts[j - 1]], 1.0)
        d_route = np.empty((len(A), len(B)))
        for ia, a in enumerate(A):
            rest = alen[a] - sA[ia]
            cache = graph._cache[int(to_node[a])][1]
            for ib, b in enumerate(B):
                if a == b:
                    d = sB[ib] - sA[ia]
                    d_route[ia, ib] = abs(d) if d >= -BACKWARD_TOL_M else BACKWARD_FACTOR * abs(d)
                else:
                    u, v = int(to_node[a]), int(from_node[b])
                    sp = 0.0 if u == v else cache.get(v, np.inf)
                    d_route[ia, ib] = rest + sp + sB[ib]
        beta_eff = beta * np.sqrt(dt)
        trans = -np.abs(d_route - d_gps) / beta_eff
        trans = np.where(np.isfinite(trans), np.maximum(trans, -max_penalty), -max_penalty)
        score = delta[:, None] + trans
        back = np.argmax(score, axis=0)
        delta = score[back, np.arange(len(B))] + em[pb]
        backs.append(back)

    k = int(np.argmax(delta))
    states = np.empty(len(pts), dtype=np.int64)
    states[-1] = k
    for j in range(len(pts) - 1, 0, -1):
        k = backs[j - 1][k]
        states[j - 1] = k
    pred[pts] = arc[start + states]
    decoded[pts] = True
    # forward-fill points without candidates (then back-fill the start)
    last = -1
    for i in range(n):
        if pred[i] >= 0:
            last = pred[i]
        elif last >= 0:
            pred[i] = last
    if pred[0] < 0 and decoded.any():
        pred[:np.argmax(decoded)] = pred[np.argmax(decoded)]
    return pred, decoded


def classifier_only(trip, cands, proba):
    """Independent per-point argmax (no sequence model)."""
    n = trip.n_points
    pred = np.full(n, -1, dtype=np.int64)
    if len(cands) == 0:
        return pred
    df = cands[["point_idx", "arc_id"]].assign(p=proba)
    best = df.loc[df.groupby("point_idx").p.idxmax()]
    pred[best.point_idx.to_numpy()] = best.arc_id.to_numpy()
    return pred


def route_sequence(pred):
    """Collapse per-point arcs into the ordered list of arcs driven."""
    p = pred[pred >= 0]
    if p.size == 0:
        return p
    keep = np.r_[True, p[1:] != p[:-1]]
    return p[keep]

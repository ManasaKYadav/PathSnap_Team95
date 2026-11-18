"""Candidate generation: which road arcs could each GPS point belong to?

Each arc is sampled every SAMPLE_STEP metres and the samples go into a KD-tree.
For each GPS point we query samples within radius + step/2, compute the exact
point-to-segment distance for the arcs found, keep arcs within the radius and
take the k closest. Returned arc ids are the original (trip-local) arc ids.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import CANDIDATE_RADIUS_M, MAX_CANDIDATES
from Reet.geometry import point_segment_distance

SAMPLE_STEP = 10.0


def _arc_samples(trip):
    arcs = trip.active_arcs
    L = trip.arc_len[arcs]
    n = np.maximum(2, np.ceil(L / SAMPLE_STEP).astype(int) + 1)
    owner = np.repeat(arcs, n)
    # position 0..1 along the arc for each sample
    starts = np.r_[0, np.cumsum(n)[:-1]]
    k = np.arange(owner.size) - np.repeat(starts, n)
    t = k / np.repeat(n - 1, n)
    sx = trip.ax[owner] + t * (trip.bx[owner] - trip.ax[owner])
    sy = trip.ay[owner] + t * (trip.by[owner] - trip.ay[owner])
    return np.c_[sx, sy], owner


def generate_candidates(trip, radius=CANDIDATE_RADIUS_M, k=MAX_CANDIDATES):
    """DataFrame with one row per (point_idx, arc_id): dist (m), frac (0..1), rank."""
    pts = trip.track[["x", "y"]].to_numpy()
    samples, owner = _arc_samples(trip)
    if len(samples) == 0:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist", "frac", "rank"])
    tree = cKDTree(samples)
    hits = tree.query_ball_point(pts, r=radius + SAMPLE_STEP / 2 + 1.0)
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if lens.sum() == 0:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist", "frac", "rank"])
    p_idx = np.repeat(np.arange(len(pts)), lens)
    a_idx = owner[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])]
    key = np.unique(p_idx * (trip.arcs.shape[0] + 1) + a_idx)
    p_idx, a_idx = key // (trip.arcs.shape[0] + 1), key % (trip.arcs.shape[0] + 1)
    d, t = point_segment_distance(pts[p_idx, 0], pts[p_idx, 1],
                                  trip.ax[a_idx], trip.ay[a_idx], trip.bx[a_idx], trip.by[a_idx])
    ok = d <= radius
    p_idx, a_idx, d, t = p_idx[ok], a_idx[ok], d[ok], t[ok]
    order = np.lexsort((d, p_idx))
    p_idx, a_idx, d, t = p_idx[order], a_idx[order], d[order], t[order]
    first = np.r_[0, np.flatnonzero(np.diff(p_idx)) + 1]
    rank = np.arange(len(p_idx)) - np.repeat(first, np.diff(np.r_[first, len(p_idx)]))
    keep = rank < k
    return pd.DataFrame({
        "point_idx": p_idx[keep].astype(np.int32),
        "arc_id": a_idx[keep].astype(np.int64),
        "dist": d[keep].astype(np.float32),
        "frac": t[keep].astype(np.float32),
        "rank": rank[keep].astype(np.int16),
    })

"""Feature engineering for (GPS point, candidate arc) pairs.

The model never sees arc ids (they are trip-local); it only sees geometry and
motion features that describe how well a candidate explains the observation.
"""
import numpy as np
import pandas as pd

from Reet.candidates import generate_candidates
from Reet.geometry import angle_diff_deg, bearing_deg, point_segment_distance

WINDOW = 3            # neighbouring points on each side used for the window feature
MIN_MOVE_M = 3.0      # below this, the GPS heading is unreliable

FEATURE_COLUMNS = [
    "dist",            # perpendicular distance point -> arc (m)
    "rank",            # 0 = closest candidate
    "dist_minus_min",  # dist - distance of the closest candidate (m)
    "dist_ratio",      # dist / (closest dist + 1)
    "n_cands",         # number of candidates for this point
    "heading_diff",    # |GPS heading - arc bearing| in degrees (0..180)
    "heading_valid",   # 1 if the vehicle moved enough for a heading
    "speed",           # m/s around the point
    "dt_prev",         # seconds since previous fix (clipped)
    "arc_len",         # arc length (m)
    "frac",            # position of the projection along the arc (0..1)
    "win_dist_mean",   # mean distance of the +-3 neighbouring points to this arc
    "win_dist_max",    # max distance of those neighbours to this arc
    "from_in_deg",     # number of arcs entering the arc's start node
    "to_out_deg",      # number of arcs leaving the arc's end node
]


def point_motion(trip):
    """Per-point heading, speed and time gap from neighbouring fixes."""
    x, y, t = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    n = len(x)
    i0 = np.clip(np.arange(n) - 2, 0, n - 1)
    i1 = np.clip(np.arange(n) + 2, 0, n - 1)
    dx, dy = x[i1] - x[i0], y[i1] - y[i0]
    move = np.hypot(dx, dy)
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    heading = bearing_deg(dx, dy)
    valid = move >= MIN_MOVE_M
    speed = np.where(i1 > i0, move / dt, 0.0)
    dt_prev = np.r_[1.0, np.diff(t)]
    return heading, valid, speed, np.clip(dt_prev, 0, 120)


def build_features(trip, cands=None):
    """Return candidate DataFrame with FEATURE_COLUMNS added."""
    if cands is None:
        cands = generate_candidates(trip)
    c = cands.copy()
    if c.empty:
        for col in FEATURE_COLUMNS:
            c[col] = pd.Series(dtype=np.float32)
        c["from_node"] = pd.Series(dtype=np.int64)
        c["to_node"] = pd.Series(dtype=np.int64)
        return c
    p = c["point_idx"].to_numpy()
    a = c["arc_id"].to_numpy()
    d = c["dist"].to_numpy()

    grp = c.groupby("point_idx")["dist"]
    dmin = grp.transform("min").to_numpy()
    c["dist_minus_min"] = (d - dmin).astype(np.float32)
    c["dist_ratio"] = (d / (dmin + 1.0)).astype(np.float32)
    c["n_cands"] = grp.transform("size").astype(np.int16)

    heading, valid, speed, dt_prev = point_motion(trip)
    hd = angle_diff_deg(heading[p], trip.arc_bearing[a])
    c["heading_diff"] = np.where(valid[p], hd, 90.0).astype(np.float32)  # 90 = uninformative
    c["heading_valid"] = valid[p].astype(np.int8)
    c["speed"] = speed[p].astype(np.float32)
    c["dt_prev"] = dt_prev[p].astype(np.float32)
    c["arc_len"] = trip.arc_len[a].astype(np.float32)

    # temporal context: how well does this arc explain the neighbouring points too?
    x, y = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    n = len(x)
    acc = np.zeros(len(c))
    mx = np.zeros(len(c))
    cnt = 0
    for off in range(-WINDOW, WINDOW + 1):
        q = np.clip(p + off, 0, n - 1)
        dd, _ = point_segment_distance(x[q], y[q], trip.ax[a], trip.ay[a], trip.bx[a], trip.by[a])
        acc += dd
        mx = np.maximum(mx, dd)
        cnt += 1
    c["win_dist_mean"] = (acc / cnt).astype(np.float32)
    c["win_dist_max"] = mx.astype(np.float32)

    n_nodes = trip.node_xy.shape[0]
    in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
    c["from_node"] = trip.arcs[a, 0]
    c["to_node"] = trip.arcs[a, 1]
    c["from_in_deg"] = in_deg[c["from_node"].to_numpy()].astype(np.int16)
    c["to_out_deg"] = out_deg[c["to_node"].to_numpy()].astype(np.int16)
    return c

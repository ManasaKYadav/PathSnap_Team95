"""Per-point labels from the ordered reference route.

The .route file lists the arcs driven, in order, but does not say which GPS point
belongs to which arc. We assign each point to one route position with a
monotone alignment (dynamic programming):

    cost(i, j) = dist(point i, route arc j) + min_{k <= j} cost(i-1, k)

so the matched route position never moves backwards in time, and the total
point-to-arc distance is minimal. This handles loops (the same arc twice) and
parallel roads better than "nearest route arc" per point.

Limitations: the route itself may be imperfect, and at junctions a point between
two consecutive arcs can be assigned to either; label_dist records how far each
point is from its labelled arc so unreliable labels can be inspected.
"""
import numpy as np

from Reet.geometry import point_segment_distance


def route_distance_matrix(trip, chunk=2000):
    r = trip.route
    ax, ay, bx, by = trip.ax[r], trip.ay[r], trip.bx[r], trip.by[r]
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    D = np.empty((len(px), len(r)), dtype=np.float32)
    for s in range(0, len(px), chunk):
        d, _ = point_segment_distance(px[s:s + chunk, None], py[s:s + chunk, None], ax, ay, bx, by)
        D[s:s + chunk] = d
    return D


def align_route_labels(trip):
    """Return (label_arc, label_route_pos, label_dist) arrays, one per GPS point."""
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"trip {trip.trip_id} has no route")
    D = route_distance_matrix(trip)
    n, m = D.shape
    back = np.empty((n, m), dtype=np.int32)
    idx = np.arange(m, dtype=np.int32)
    cost = D[0].astype(np.float64)
    back[0] = idx
    for i in range(1, n):
        run_min = np.minimum.accumulate(cost)
        # position of the running minimum (last position achieving it)
        arg = np.maximum.accumulate(np.where(cost == run_min, idx, 0))
        back[i] = arg
        cost = D[i] + run_min
    pos = np.empty(n, dtype=np.int32)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    label_arc = trip.route[pos]
    label_dist = D[np.arange(n), pos]
    return label_arc, pos, label_dist

"""Route-level evaluation metrics.

pointwise_accuracy  share of reliably labelled GPS points whose predicted arc equals the labelled arc
route_continuity    share of consecutive GPS points whose predicted arcs are the same or connected
route P / R / F1    length-weighted overlap between the predicted arc sequence and the reference
                    route: precision = length of predicted arcs that are on the reference route /
                    total predicted length; recall = length of reference arcs that were predicted /
                    total reference length.
"""
import numpy as np

from Rekha.hmm import route_sequence


def pointwise_accuracy(pred, label_arc, reliable):
    if reliable.sum() == 0:
        return np.nan
    return float(np.mean(pred[reliable] == label_arc[reliable]))


def route_continuity(trip, pred):
    p = pred[pred >= 0]
    if p.size < 2:
        return np.nan
    same = p[1:] == p[:-1]
    linked = trip.arcs[p[:-1], 1] == trip.arcs[p[1:], 0]
    return float(np.mean(same | linked))


def route_overlap(trip, pred, ref_route):
    seq = np.unique(route_sequence(pred))
    ref = np.unique(ref_route)
    L = trip.arc_len
    inter = np.intersect1d(seq, ref)
    prec = L[inter].sum() / max(L[seq].sum(), 1e-9)
    rec = L[inter].sum() / max(L[ref].sum(), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return float(prec), float(rec), float(f1)


def all_metrics(trip, pred, label_arc, reliable):
    prec, rec, f1 = route_overlap(trip, pred, trip.route)
    return dict(pointwise_accuracy=pointwise_accuracy(pred, label_arc, reliable),
                route_continuity=route_continuity(trip, pred),
                route_precision=prec, route_recall=rec, route_f1=f1,
                reliable_points=int(reliable.sum()), points=int(len(pred)))

"""End-to-end prediction pipeline for one trip folder.

load trip (WITHOUT the route file) -> candidates + features (Reet) -> classifier
probabilities (Manasa) -> HMM/Viterbi (Rekha) -> predicted arc per GPS point.

Ground truth is never read here; evaluation is done separately (Sourish's API
computes accuracy server-side using Reet's label alignment).
"""
import json
import time

import numpy as np

from config import DECODER_CONFIG
from Manasa.predict import load_model, predict_proba
from Reet.data_loader import load_trip
from Reet.features import build_features
from Rekha.hmm import classifier_only, emission_logp, route_sequence, viterbi_decode
from Rekha.road_graph import RoadGraph

DEFAULT_DECODER = {"emission": "classifier", "beta": 5.0, "emission_weight": 1.0,
                   "max_penalty": 50.0, "sigma": 10.0}


def load_decoder_config(path=DECODER_CONFIG):
    try:
        cfg = json.loads(path.read_text())
        return {**DEFAULT_DECODER, **cfg.get("best", cfg)}
    except FileNotFoundError:
        return dict(DEFAULT_DECODER)


def decode_trip(trip, cands, proba, cfg):
    graph = RoadGraph(trip)
    if cfg["emission"] == "distance":
        em = emission_logp(cands, "distance", sigma=cfg["sigma"])
    else:
        em = emission_logp(cands, "classifier", proba=proba, weight=cfg["emission_weight"])
    return viterbi_decode(trip, graph, cands, em, beta=cfg["beta"], max_penalty=cfg["max_penalty"])


def run_pipeline(trip_folder, bundle=None, cfg=None):
    timings = {}
    t0 = time.perf_counter()
    bundle = bundle or load_model()
    cfg = cfg or load_decoder_config()
    trip = load_trip(trip_folder, with_route=False)
    timings["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    cands = build_features(trip)
    timings["features_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    proba = predict_proba(bundle, cands)
    timings["classifier_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    pred, decoded = decode_trip(trip, cands, proba, cfg)
    timings["viterbi_s"] = time.perf_counter() - t1
    timings["total_s"] = time.perf_counter() - t0

    seq = route_sequence(pred)
    return {
        "trip": trip,
        "pred_arcs": pred,                       # one arc id per GPS point (-1 = none)
        "decoded": decoded,                      # False where the point had no candidates
        "classifier_only_arcs": classifier_only(trip, cands, proba),
        "route_arcs": seq,                       # ordered arcs driven
        "timings": timings,
        "n_candidates": int(len(cands)),
    }


def route_lonlat(trip, seq):
    """[[lon, lat], ...] polyline through the arcs of `seq`."""
    pts = []
    for a in seq:
        u, v = trip.arcs[int(a)]
        for node in (u, v):
            p = [float(trip.node_lonlat[node, 0]), float(trip.node_lonlat[node, 1])]
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts

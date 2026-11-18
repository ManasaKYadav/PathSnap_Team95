"""Tune the HMM decoder parameters on TRAINING data only.

To avoid tuning on trips the classifier was fitted on, 10 training trips are held
out: a copy of the final classifier (same hyperparameters) is refitted on the
other 70 training trips and the decoder grid is evaluated on the 10 held-out trips.

Grid
  classifier emission: beta x emission_weight
  distance emission (baseline HMM without the classifier): beta x sigma
Selection: highest point-weighted pointwise accuracy; ties broken by route F1.

Outputs: Rekha/results/decoder_grid.csv, Rekha/decoder_config.json
Run: python -m Rekha.tune_decoder
"""
import json
import itertools

import numpy as np
import pandas as pd

from config import DECODER_CONFIG, LABEL_MAX_DIST_M, REKHA_RESULTS, SEED, TRAIN_DIR
from Manasa.data import X, load_train
from Manasa.models import MODELS
from Manasa.predict import load_model, predict_proba
from Reet.data_loader import list_trips, load_trip
from Reet.features import build_features
from Reet.labels import align_route_labels
from Rekha.hmm import classifier_only, emission_logp, viterbi_decode
from Rekha.road_graph import RoadGraph
from Rekha.route_metrics import all_metrics

BETAS = [1, 2, 5, 10, 20, 50]
WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]
SIGMAS = [3, 5, 10, 20]
MAX_PENALTY = 50.0


def main():
    REKHA_RESULTS.mkdir(parents=True, exist_ok=True)
    final = load_model()
    folders = list_trips(TRAIN_DIR)
    rng = np.random.default_rng(SEED)
    hold = sorted(str(x) for x in rng.choice([f.name for f in folders], size=10, replace=False))
    print("held-out training trips:", hold)

    train, _ = load_train()
    train = train[~train.trip_id.isin(hold)]
    model = MODELS[final["model_name"]][2](**final["params"])
    model.fit(X(train), train.label)
    bundle = dict(model=model, feature_columns=final["feature_columns"])

    data = []
    for tid in hold:
        trip = load_trip(TRAIN_DIR / tid)
        la, _, ld = align_route_labels(trip)
        cands = build_features(trip)
        data.append((trip, RoadGraph(trip), cands, predict_proba(bundle, cands), la, ld <= LABEL_MAX_DIST_M))

    rows = []
    accs, f1s, conts, npts = [], [], [], []
    for trip, graph, cands, proba, la, rel in data:  # reference: no HMM
        m = all_metrics(trip, classifier_only(trip, cands, proba), la, rel)
        accs.append(m["pointwise_accuracy"]); f1s.append(m["route_f1"])
        conts.append(m["route_continuity"]); npts.append(m["reliable_points"])
    rows.append(dict(emission="classifier_only (no HMM)", beta=None, emission_weight=None, sigma=None,
                     pointwise_accuracy=np.average(accs, weights=npts), route_f1=np.mean(f1s),
                     route_continuity=np.mean(conts)))
    grid = [("classifier", b, w, None) for b, w in itertools.product(BETAS, WEIGHTS)] + \
           [("distance", b, None, s) for b, s in itertools.product(BETAS, SIGMAS)]
    for mode, beta, w, sigma in grid:
        accs, f1s, conts, npts = [], [], [], []
        for trip, graph, cands, proba, la, rel in data:
            em = emission_logp(cands, mode, proba=proba, weight=w or 1.0, sigma=sigma or 10.0)
            pred, _ = viterbi_decode(trip, graph, cands, em, beta=beta, max_penalty=MAX_PENALTY)
            m = all_metrics(trip, pred, la, rel)
            accs.append(m["pointwise_accuracy"]); f1s.append(m["route_f1"])
            conts.append(m["route_continuity"]); npts.append(m["reliable_points"])
        rows.append(dict(emission=mode, beta=beta, emission_weight=w, sigma=sigma,
                         pointwise_accuracy=np.average(accs, weights=npts), route_f1=np.mean(f1s),
                         route_continuity=np.mean(conts)))
        print(rows[-1])
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(REKHA_RESULTS / "decoder_grid.csv", index=False)

    def best_of(mode):
        g = grid_df[grid_df.emission == mode].sort_values(["pointwise_accuracy", "route_f1"], ascending=False)
        return g.iloc[0]

    bc, bd = best_of("classifier"), best_of("distance")
    cfg = {
        "best": {"emission": "classifier", "beta": float(bc.beta), "emission_weight": float(bc.emission_weight),
                 "max_penalty": MAX_PENALTY, "sigma": float(bd.sigma)},
        "distance_baseline": {"emission": "distance", "beta": float(bd.beta), "sigma": float(bd.sigma),
                              "max_penalty": MAX_PENALTY, "emission_weight": 1.0},
        "tuned_on": f"10 held-out training trips {hold}; classifier refitted on the other 70",
        "selection": "highest point-weighted pointwise accuracy, ties by route F1",
    }
    DECODER_CONFIG.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

"""Compare route reconstruction methods on the 10 student-testing trips.

  1. classifier-only        : per-point argmax of the final classifier (no sequence model)
  2. HMM, distance emission : Viterbi with a Gaussian distance emission (no classifier)
  3. HMM, classifier emission (final system)

Outputs (Rekha/results/): hmm_comparison_per_trip.csv, hmm_comparison_summary.csv,
  hmm_changes.csv (points fixed / broken by the HMM), plots/*.png

Run: python -m Rekha.evaluate_hmm
"""
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DECODER_CONFIG, LABEL_MAX_DIST_M, REKHA_RESULTS, STUDENT_TEST_DIR
from Manasa.predict import load_model, predict_proba
from Reet.data_loader import list_trips, load_trip
from Reet.features import build_features
from Reet.labels import align_route_labels
from Rekha.hmm import classifier_only, emission_logp, viterbi_decode
from Rekha.road_graph import RoadGraph
from Rekha.route_metrics import all_metrics

PLOTS = REKHA_RESULTS / "plots"
METHODS = ["classifier_only", "hmm_distance", "hmm_classifier"]


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(DECODER_CONFIG.read_text())
    best, dist_cfg = cfg["best"], cfg["distance_baseline"]
    bundle = load_model()
    rows, changes = [], []
    for folder in list_trips(STUDENT_TEST_DIR):
        trip = load_trip(folder)
        la, _, ld = align_route_labels(trip)
        rel = ld <= LABEL_MAX_DIST_M
        cands = build_features(trip)
        proba = predict_proba(bundle, cands)
        graph = RoadGraph(trip)
        preds, times = {}, {}
        t0 = time.perf_counter()
        preds["classifier_only"] = classifier_only(trip, cands, proba)
        times["classifier_only"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        em = emission_logp(cands, "distance", sigma=dist_cfg["sigma"])
        preds["hmm_distance"], _ = viterbi_decode(trip, graph, cands, em, dist_cfg["beta"], dist_cfg["max_penalty"])
        times["hmm_distance"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        em = emission_logp(cands, "classifier", proba=proba, weight=best["emission_weight"])
        preds["hmm_classifier"], _ = viterbi_decode(trip, graph, cands, em, best["beta"], best["max_penalty"])
        times["hmm_classifier"] = time.perf_counter() - t0
        for meth in METHODS:
            m = all_metrics(trip, preds[meth], la, rel)
            rows.append(dict(trip_id=trip.trip_id, method=meth, decode_time_s=times[meth], **m))
        c_ok = preds["classifier_only"] == la
        h_ok = preds["hmm_classifier"] == la
        changes.append(dict(trip_id=trip.trip_id, reliable_points=int(rel.sum()),
                            fixed_by_hmm=int((~c_ok & h_ok & rel).sum()),
                            broken_by_hmm=int((c_ok & ~h_ok & rel).sum())))
        np.save(REKHA_RESULTS / f"pred_{trip.trip_id}.npy", np.vstack([preds[m] for m in METHODS]))
    per = pd.DataFrame(rows)
    per.to_csv(REKHA_RESULTS / "hmm_comparison_per_trip.csv", index=False)
    pd.DataFrame(changes).to_csv(REKHA_RESULTS / "hmm_changes.csv", index=False)

    def wavg(g):
        return pd.Series(dict(
            pointwise_accuracy=np.average(g.pointwise_accuracy, weights=g.reliable_points),
            route_continuity=g.route_continuity.mean(), route_precision=g.route_precision.mean(),
            route_recall=g.route_recall.mean(), route_f1=g.route_f1.mean(),
            decode_time_s_total=g.decode_time_s.sum()))
    summ = per.groupby("method").apply(wavg).loc[METHODS]
    summ.to_csv(REKHA_RESULTS / "hmm_comparison_summary.csv")
    print(summ)
    print(pd.DataFrame(changes))

    fig, ax = plt.subplots(figsize=(8, 4))
    mets = ["pointwise_accuracy", "route_continuity", "route_f1"]
    xs = np.arange(len(mets))
    for i, meth in enumerate(METHODS):
        ax.bar(xs + (i - 1) * 0.27, summ.loc[meth, mets], 0.27, label=meth)
    ax.set_xticks(xs, mets)
    ax.set_ylim(0.7, 1.0)
    ax.set(title="Classifier-only vs HMM decoding (student-test, 10 trips)", ylabel="score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "hmm_vs_classifier.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    piv = per.pivot(index="trip_id", columns="method", values="pointwise_accuracy")[METHODS]
    piv.plot.bar(ax=ax, width=0.8)
    ax.set_ylim(0.5, 1.0)
    ax.set(title="Pointwise accuracy per student-test trip", ylabel="accuracy")
    fig.tight_layout()
    fig.savefig(PLOTS / "pointwise_accuracy_per_trip.png", dpi=150)
    plt.close(fig)

    grid = pd.read_csv(REKHA_RESULTS / "decoder_grid.csv")
    g = grid[grid.emission == "classifier"].pivot(index="emission_weight", columns="beta", values="pointwise_accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(g.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(g.columns)), g.columns)
    ax.set_yticks(range(len(g.index)), g.index)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set(xlabel="beta (m)", ylabel="emission weight",
           title="Decoder grid: pointwise accuracy (10 held-out training trips)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(PLOTS / "decoder_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

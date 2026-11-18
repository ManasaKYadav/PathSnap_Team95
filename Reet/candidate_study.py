"""Choose candidate radius / k by measuring the candidate miss rate on TRAINING trips.

Miss rate = share of reliably labelled GPS points whose labelled arc is not in their candidate set.
Run: python -m Reet.candidate_study
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import LABEL_MAX_DIST_M, REET_RESULTS, TRAIN_DIR
from Reet.candidates import generate_candidates
from Reet.data_loader import list_trips, load_trip
from Reet.labels import align_route_labels

RADII = [20, 30, 50, 80]
KS = [3, 5, 8, 12]


def main():
    REET_RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in list_trips(TRAIN_DIR):
        trip = load_trip(folder)
        label, _, label_dist = align_route_labels(trip)
        reliable = label_dist <= LABEL_MAX_DIST_M
        cands = generate_candidates(trip, radius=max(RADII), k=max(KS))
        cands = cands[reliable[cands.point_idx.to_numpy()]]
        cands["is_true"] = cands.arc_id.to_numpy() == label[cands.point_idx.to_numpy()]
        n_rel = int(reliable.sum())
        for r in RADII:
            for k in KS:
                sub = cands[(cands.dist <= r) & (cands["rank"] < k)]
                hit = sub.groupby("point_idx").is_true.any()
                n_hit = int(hit.sum())
                rows.append(dict(trip=trip.trip_id, radius_m=r, k=k, points=n_rel,
                                 hits=n_hit, mean_cands=len(sub) / max(n_rel, 1)))
    df = pd.DataFrame(rows)
    summ = df.groupby(["radius_m", "k"]).agg(points=("points", "sum"), hits=("hits", "sum"),
                                            mean_cands=("mean_cands", "mean")).reset_index()
    summ["miss_rate"] = 1 - summ.hits / summ.points
    summ.to_csv(REET_RESULTS / "candidate_miss_rate.csv", index=False)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in RADII:
        s = summ[summ.radius_m == r]
        ax.plot(s.k, 100 * s.miss_rate, marker="o", label=f"radius {r} m")
    ax.set_xlabel("max candidates per point (k)")
    ax.set_ylabel("candidate miss rate (%)")
    ax.set_title("Candidate miss rate, training partition (80 trips)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REET_RESULTS / "candidate_miss_rate.png", dpi=150)


if __name__ == "__main__":
    main()

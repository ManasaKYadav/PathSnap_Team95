"""Build the model-ready datasets for the training and student-testing partitions.

Outputs (Reet/output/):
  train_candidates.parquet, student_test_candidates.parquet
      one row per (trip_id, point_idx, candidate arc): features + label (1 = true arc)
  train_points.parquet, student_test_points.parquet
      one row per GPS point: label arc, label distance, reliability, candidate hit
  build_log.csv  per-trip cleaning and size statistics

Run: python -m Reet.build_dataset
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import LABEL_MAX_DIST_M, REET_OUT, STUDENT_TEST_DIR, TRAIN_DIR
from Reet.data_loader import list_trips, load_trip
from Reet.features import FEATURE_COLUMNS, build_features
from Reet.labels import align_route_labels


def process_trip(folder):
    trip = load_trip(folder)
    label_arc, _, label_dist = align_route_labels(trip)
    reliable = label_dist <= LABEL_MAX_DIST_M
    feats = build_features(trip)
    p = feats.point_idx.to_numpy()
    feats["label"] = (feats.arc_id.to_numpy() == label_arc[p]).astype(np.int8)
    feats["reliable"] = reliable[p]
    feats.insert(0, "trip_id", trip.trip_id)

    hit = feats.groupby("point_idx").label.max().reindex(range(trip.n_points), fill_value=0).to_numpy()
    ncand = feats.groupby("point_idx").size().reindex(range(trip.n_points), fill_value=0).to_numpy()
    points = pd.DataFrame({
        "trip_id": trip.trip_id,
        "point_idx": np.arange(trip.n_points, dtype=np.int32),
        "t": trip.track.t.to_numpy(),
        "lon": trip.track.lon.to_numpy(),
        "lat": trip.track.lat.to_numpy(),
        "label_arc": label_arc,
        "label_dist": label_dist,
        "reliable": reliable,
        "has_true_cand": hit.astype(bool),
        "n_cands": ncand,
    })
    log = dict(trip_id=trip.trip_id, **trip.cleaning, arcs=len(trip.arcs), active_arcs=len(trip.active_arcs),
               route_arcs=len(trip.route), candidate_rows=len(feats), reliable_points=int(reliable.sum()),
               miss_rate_reliable=float(1 - hit[reliable].mean()) if reliable.any() else np.nan)
    return feats, points, log


def build_partition(name, part_dir, workers=4):
    folders = list_trips(part_dir)
    with Pool(workers) as pool:
        out = pool.map(process_trip, folders)
    feats = pd.concat([o[0] for o in out], ignore_index=True)
    points = pd.concat([o[1] for o in out], ignore_index=True)
    logs = pd.DataFrame([o[2] for o in out])
    logs.insert(0, "partition", name)
    feats.to_parquet(REET_OUT / f"{name}_candidates.parquet", index=False)
    points.to_parquet(REET_OUT / f"{name}_points.parquet", index=False)
    print(f"{name}: {len(folders)} trips, {len(points)} points, {len(feats)} candidate rows, "
          f"positive rate {feats.label.mean():.3f}")
    return logs


def main():
    REET_OUT.mkdir(parents=True, exist_ok=True)
    logs = pd.concat([build_partition("train", TRAIN_DIR), build_partition("student_test", STUDENT_TEST_DIR)])
    logs.to_csv(REET_OUT / "build_log.csv", index=False)
    print("features:", FEATURE_COLUMNS)


if __name__ == "__main__":
    main()

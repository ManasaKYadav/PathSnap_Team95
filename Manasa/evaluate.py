"""Evaluation metrics for candidate classifiers.

Row level (each candidate row is a binary example): accuracy, precision, recall,
Macro F1, log loss, confusion matrix.

Observation level:
  Top-1 accuracy   for each reliably labelled GPS point, the candidate with the
                   highest probability is compared with the labelled arc. Points
                   whose true arc is missing from the candidate set count as wrong.
  Route continuity share of consecutive GPS points whose predicted arcs are the
                   same arc or directly connected (end node of one = start node of next).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def row_metrics(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def confusion(y_true, proba, threshold=0.5):
    return confusion_matrix(y_true, (proba >= threshold).astype(int), labels=[0, 1])


def top1_predictions(cands: pd.DataFrame, proba):
    """Per (trip, point): the arc with the highest score. Returns DataFrame."""
    df = cands[["trip_id", "point_idx", "arc_id", "from_node", "to_node", "label"]].copy()
    df["proba"] = proba
    idx = df.groupby(["trip_id", "point_idx"])["proba"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def top1_accuracy(cands, proba, n_points):
    """n_points = number of reliable GPS points evaluated (includes points with no candidates)."""
    best = top1_predictions(cands, proba)
    return best.label.sum() / n_points


def route_continuity(best: pd.DataFrame):
    """best: one row per point (sorted by trip, point_idx) with arc_id, from_node, to_node."""
    best = best.sort_values(["trip_id", "point_idx"])
    same_trip = best.trip_id.to_numpy()[1:] == best.trip_id.to_numpy()[:-1]
    a = best.arc_id.to_numpy()
    same = a[1:] == a[:-1]
    linked = best.to_node.to_numpy()[:-1] == best.from_node.to_numpy()[1:]
    ok = (same | linked)[same_trip]
    return float(ok.mean()) if ok.size else np.nan

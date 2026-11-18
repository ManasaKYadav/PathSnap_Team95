"""Load Reet's processed datasets for modelling."""
import pandas as pd

from config import REET_OUT, TRAIN_POINT_STRIDE
from Reet.features import FEATURE_COLUMNS


def load_train(stride=TRAIN_POINT_STRIDE):
    """Training rows: reliable points only, every `stride`-th GPS point per trip.
    Returns (candidate rows, number of reliable points per trip after striding)."""
    c = pd.read_parquet(REET_OUT / "train_candidates.parquet")
    p = pd.read_parquet(REET_OUT / "train_points.parquet")
    c = c[c.reliable & (c.point_idx % stride == 0)].reset_index(drop=True)
    p = p[p.reliable & (p.point_idx % stride == 0)]
    return c, p.groupby("trip_id").size()


def load_student_test():
    """All reliable student-test rows (no striding)."""
    c = pd.read_parquet(REET_OUT / "student_test_candidates.parquet")
    p = pd.read_parquet(REET_OUT / "student_test_points.parquet")
    c = c[c.reliable].reset_index(drop=True)
    p = p[p.reliable]
    return c, p.groupby("trip_id").size()


def X(df):
    return df[FEATURE_COLUMNS].to_numpy(dtype="float32")

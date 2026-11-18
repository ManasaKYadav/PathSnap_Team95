"""Load the final model and score candidate rows (used by Rekha's pipeline and the API)."""
import joblib
import numpy as np

from config import MODEL_DIR


def load_model(path=MODEL_DIR / "final_model.joblib"):
    return joblib.load(path)


def predict_proba(bundle, cands):
    """P(candidate is the true arc) for each row of a candidate feature DataFrame."""
    if len(cands) == 0:
        return np.zeros(0)
    Xm = cands[bundle["feature_columns"]].to_numpy(dtype="float32")
    return bundle["model"].predict_proba(Xm)[:, 1]

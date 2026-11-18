"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            status, loaded model and decoder settings
  GET  /trips             trip ids available in the evaluation folder
  POST /predict           {"trip_id": "..."} -> runs the full pipeline on that trip and
                          returns the predicted route + accuracy computed SERVER-SIDE

Ground truth (.route) is read only here, after prediction, to compute metrics. It is
never passed to the model and never returned to the UI.

Run (from the project root):
  uvicorn Sourish.api.main:app --host 0.0.0.0 --port 8000
Evaluation folder: env var PATHSNAP_EVAL_DIR (default: data/raw/student_test).
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import LABEL_MAX_DIST_M, STUDENT_TEST_DIR
from Manasa.predict import load_model
from Reet.data_loader import list_trips, load_trip
from Reet.labels import align_route_labels
from Rekha.pipeline import load_decoder_config, route_lonlat, run_pipeline
from Rekha.route_metrics import all_metrics

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", STUDENT_TEST_DIR))
MAX_GPS_POINTS_RETURNED = 3000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {}


@app.on_event("startup")
def startup():
    STATE["bundle"] = load_model()
    STATE["decoder"] = load_decoder_config()
    log.info("model loaded: %s | decoder: %s", STATE["bundle"]["model_name"], STATE["decoder"])
    log.info("evaluation folder: %s (%d trips)", EVAL_DIR, len(trip_ids()))


def trip_ids():
    return [p.name for p in list_trips(EVAL_DIR)] if EVAL_DIR.exists() else []


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.perf_counter() - t0)
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["bundle"]["model_name"], "decoder": STATE["decoder"],
            "eval_dir": str(EVAL_DIR), "trips": len(trip_ids())}


@app.get("/trips")
def trips():
    ids = trip_ids()
    return {"count": len(ids), "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in trip_ids():
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id}")
    folder = EVAL_DIR / req.trip_id
    log.info("predict request: trip %s", req.trip_id)

    out = run_pipeline(folder, STATE["bundle"], STATE["decoder"])
    trip = out["trip"]  # loaded without the route
    pred = out["pred_arcs"]

    # ---- server-side evaluation (ground truth stays here) ----
    metrics = {}
    gt_trip = load_trip(folder, with_route=True)
    if gt_trip.route is not None:
        label, _, label_dist = align_route_labels(gt_trip)
        reliable = label_dist <= LABEL_MAX_DIST_M
        hmm = all_metrics(gt_trip, pred, label, reliable)
        clf = all_metrics(gt_trip, out["classifier_only_arcs"], label, reliable)
        metrics = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in hmm.items()},
                   "classifier_only_accuracy": round(clf["pointwise_accuracy"], 4),
                   "classifier_only_continuity": round(clf["route_continuity"], 4)}
    log.info("trip %s | points %d | accuracy %s | continuity %s | %.2fs", req.trip_id, trip.n_points,
             metrics.get("pointwise_accuracy"), metrics.get("route_continuity"), out["timings"]["total_s"])

    step = max(1, trip.n_points // MAX_GPS_POINTS_RETURNED)
    gps = trip.track.iloc[::step]
    return {
        "trip_id": req.trip_id,
        "n_gps_points": trip.n_points,
        "n_points_without_candidates": int((~out["decoded"]).sum()),
        "n_route_arcs": int(len(out["route_arcs"])),
        "predicted_route": [[lat, lon] for lon, lat in route_lonlat(trip, out["route_arcs"])],
        "gps_points": np.c_[gps.lat.to_numpy(), gps.lon.to_numpy()].round(6).tolist(),
        "metrics": metrics,
        "timings_s": {k: round(v, 3) for k, v in out["timings"].items()},
        "model": STATE["bundle"]["model_name"],
    }

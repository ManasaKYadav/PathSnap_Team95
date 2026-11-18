"""Shared project configuration (paths, seeds, candidate settings).

Every member imports from here so all parts of the pipeline use the same values.
Paths can be overridden with environment variables.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Raw data: data/raw/train/<trip>/..., data/raw/student_test/<trip>/...
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "raw"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"

# Outputs of each member
REET_OUT = ROOT / "Reet" / "output"          # processed parquet files
REET_RESULTS = ROOT / "Reet" / "results"     # EDA tables, plots, maps
MANASA_RESULTS = ROOT / "Manasa" / "results"
MODEL_DIR = ROOT / "Manasa" / "models"
REKHA_RESULTS = ROOT / "Rekha" / "results"
DECODER_CONFIG = ROOT / "Rekha" / "decoder_config.json"

SEED = 42

# Projection: trips are on several continents, so each trip is projected to its own
# UTM zone (see Reet/geometry.py: utm_crs_for)

# Candidate generation (chosen from Reet/candidate_study.py results)
CANDIDATE_RADIUS_M = 50.0
MAX_CANDIDATES = 8

# Road network is cropped to the track's bounding box plus this buffer (metres)
CROP_BUFFER_M = 600.0

# Points used for model fitting: every Nth GPS point of each training trip
# (consecutive 1 Hz points are almost identical, so this reduces size without losing information)
TRAIN_POINT_STRIDE = 5

# Points farther than this from their labelled route arc get no reliable label
# (some reference routes cover only part of the track, e.g. trips 00000054, 00000033).
# They are excluded from training and from accuracy scoring, and reported in the EDA.
LABEL_MAX_DIST_M = 50.0

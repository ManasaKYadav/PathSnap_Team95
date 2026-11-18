"""Load and clean one PathSnap trip.

Raw files per trip folder <id>/:
  <id>.track  lon lat timestamp      (GPS observations, ~1 Hz)
  <id>.nodes  lon lat                (row index = node id)
  <id>.arcs   from_node to_node      (row index = arc id, directed)
  <id>.route  arc_id                 (ordered ground-truth arcs; may be absent)
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import CROP_BUFFER_M
from Reet.geometry import bearing_deg, project, utm_crs_for

MAX_SPEED_MS = 70.0  # ~250 km/h: anything faster between two fixes is a GPS spike


@dataclass
class Trip:
    trip_id: str
    folder: Path
    track: pd.DataFrame            # cleaned: lon, lat, t, x, y, raw_idx
    node_lonlat: np.ndarray        # (N, 2)
    node_xy: np.ndarray            # (N, 2) metres
    arcs: np.ndarray               # (M, 2) from_node, to_node
    route: np.ndarray | None       # ground-truth arc ids (None at inference)
    crs: str = ""                  # UTM zone used for this trip
    cleaning: dict = field(default_factory=dict)

    # arc geometry (filled by _arc_geometry)
    ax: np.ndarray = None
    ay: np.ndarray = None
    bx: np.ndarray = None
    by: np.ndarray = None
    arc_len: np.ndarray = None
    arc_bearing: np.ndarray = None
    active_arcs: np.ndarray = None  # arc ids inside the cropped area

    @property
    def n_points(self):
        return len(self.track)

    def _arc_geometry(self):
        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.ax, self.ay = self.node_xy[a, 0], self.node_xy[a, 1]
        self.bx, self.by = self.node_xy[b, 0], self.node_xy[b, 1]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.arc_len = np.hypot(dx, dy)
        self.arc_bearing = bearing_deg(dx, dy)
        # crop the network to the area around the track (big trips have 300k arcs)
        x, y = self.track["x"].to_numpy(), self.track["y"].to_numpy()
        x0, x1 = x.min() - CROP_BUFFER_M, x.max() + CROP_BUFFER_M
        y0, y1 = y.min() - CROP_BUFFER_M, y.max() + CROP_BUFFER_M
        inside = (np.maximum(self.ax, self.bx) >= x0) & (np.minimum(self.ax, self.bx) <= x1) & \
                 (np.maximum(self.ay, self.by) >= y0) & (np.minimum(self.ay, self.by) <= y1)
        self.active_arcs = np.flatnonzero(inside)


def _read(path, dtype=float, ndmin=2):
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def clean_track(raw: np.ndarray):
    """Remove invalid rows, sort by time, drop duplicate timestamps and speed spikes."""
    info = {"raw_points": len(raw)}
    df = pd.DataFrame(raw[:, :3], columns=["lon", "lat", "t"])
    df["raw_idx"] = np.arange(len(df))
    valid = df[["lon", "lat", "t"]].notna().all(axis=1) & df.lon.between(-180, 180) & df.lat.between(-90, 90)
    info["invalid_dropped"] = int((~valid).sum())
    df = df[valid].sort_values("t", kind="stable")
    dup = df.duplicated("t", keep="first")
    info["duplicate_time_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    crs = utm_crs_for(float(df.lon.median()), float(df.lat.median()))
    info["crs"] = crs
    x, y = project(df.lon.to_numpy(), df.lat.to_numpy(), crs)
    df["x"], df["y"] = x, y

    # drop single-point spikes: a point that is far from both neighbours at an impossible speed
    keep = np.ones(len(df), dtype=bool)
    if len(df) > 2:
        t = df.t.to_numpy()
        d_prev = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
        dt_prev = np.r_[1, np.diff(t)]
        d_next = np.r_[np.hypot(np.diff(x), np.diff(y)), 0]
        dt_next = np.r_[np.diff(t), 1]
        spike = (d_prev / dt_prev > MAX_SPEED_MS) & (d_next / dt_next > MAX_SPEED_MS)
        keep &= ~spike
    info["speed_spikes_dropped"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    info["clean_points"] = len(df)
    return df, info


def load_trip(folder, with_route=True) -> Trip:
    """Load one trip folder. with_route=False is used by the prediction pipeline,
    so ground truth never enters inference."""
    folder = Path(folder)
    tid = folder.name
    raw = _read(folder / f"{tid}.track")
    track, info = clean_track(raw)
    nodes = _read(folder / f"{tid}.nodes")
    arcs = _read(folder / f"{tid}.arcs", dtype=np.int64)
    route = None
    rpath = folder / f"{tid}.route"
    if with_route and rpath.exists():
        route = _read(rpath, dtype=np.int64, ndmin=1)
    crs = info["crs"]
    nx, ny = project(nodes[:, 0], nodes[:, 1], crs)
    trip = Trip(tid, folder, track, nodes[:, :2], np.c_[nx, ny], arcs, route, crs, info)
    trip._arc_geometry()
    return trip


def list_trips(partition_dir):
    partition_dir = Path(partition_dir)
    return sorted(p for p in partition_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())

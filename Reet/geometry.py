"""Geometry helpers: projection to metres, point-to-segment distance, bearings.

Longitude/latitude are angles, so distances in degrees are not physical distances
(one degree of longitude is 111 km at the equator but about 62 km at 56 N).
The trips are spread over several continents, so each trip is projected to its
OWN UTM zone (chosen from the trip's centre; northern zones EPSG:326xx, southern
EPSG:327xx). Inside one UTM zone x/y are metres with very small distortion.
"""
from functools import lru_cache

import numpy as np
from pyproj import Transformer


def utm_crs_for(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


@lru_cache(maxsize=None)
def _transformer(crs, inverse=False):
    if inverse:
        return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def project(lon, lat, crs):
    """lon/lat (degrees) -> x/y (metres) in the given projected CRS."""
    x, y = _transformer(crs).transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def unproject(x, y, crs):
    lon, lat = _transformer(crs, True).transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(lon), np.asarray(lat)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB.

    Returns (distance in metres, t) where t in [0, 1] is the position of the
    projection of P along AB (0 = at A, 1 = at B).
    """
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = np.where(seg_len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


def bearing_deg(dx, dy):
    """Compass bearing in degrees (0 = north, 90 = east) of a movement vector."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

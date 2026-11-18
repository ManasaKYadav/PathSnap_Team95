"""Call the running API for every trip and print / save a results table.

Usage: python -m Sourish.api.test_client [http://<laptop1-ip>:8000]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

api = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
rows = []
for tid in trips:
    t0 = time.perf_counter()
    r = requests.post(f"{api}/predict", json={"trip_id": tid}, timeout=600)
    r.raise_for_status()
    res = r.json()
    assert "route" not in res and "reference_route" not in res  # ground truth must not leave the server
    rows.append(dict(trip_id=tid, gps_points=res["n_gps_points"], route_arcs=res["n_route_arcs"],
                     **res["metrics"], response_s=round(time.perf_counter() - t0, 2)))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
out = Path(__file__).resolve().parents[1] / "results"
out.mkdir(exist_ok=True)
df.to_csv(out / "api_test_results.csv", index=False)

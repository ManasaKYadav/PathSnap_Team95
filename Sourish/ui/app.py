"""PathSnap evaluator UI (runs on Laptop 2). Talks to the API only; has no model or data.

Run:  streamlit run Sourish/ui/app.py
Set the API address in Sourish/ui/config.json (or in the sidebar).
"""
import json
import math
import time
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import streamlit as st
import streamlit.components.v1 as components

CFG = json.loads((Path(__file__).parent / "config.json").read_text())

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching and route reconstruction")

api = st.sidebar.text_input("API address", CFG.get("api_url", "http://127.0.0.1:8000")).rstrip("/")


@st.cache_data(ttl=30)
def get_trips(api_url):
    r = requests.get(f"{api_url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


try:
    trips = get_trips(api)
    health = requests.get(f"{api}/health", timeout=10).json()
    st.sidebar.success(f"Connected. Model: {health['model']}")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}: {e}")
    st.stop()

st.write(f"**{len(trips)} evaluation trips available.** Select one and run the prediction.")
trip_id = st.selectbox("Evaluation trip", trips)

if st.button("Run prediction", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        t0 = time.perf_counter()
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
        elapsed = time.perf_counter() - t0
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = (r.json(), elapsed)

if "result" in st.session_state:
    res, elapsed = st.session_state["result"]
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']}")
    c = st.columns(5)
    c[0].metric("Pointwise accuracy (HMM)", f"{m.get('pointwise_accuracy', float('nan')):.2%}")
    c[1].metric("Classifier-only accuracy", f"{m.get('classifier_only_accuracy', float('nan')):.2%}")
    c[2].metric("Route continuity", f"{m.get('route_continuity', float('nan')):.2%}")
    c[3].metric("Route F1 (length)", f"{m.get('route_f1', float('nan')):.2%}")
    c[4].metric("Response time", f"{elapsed:.2f} s")
    st.caption(f"{res['n_gps_points']} GPS points, {res['n_route_arcs']} predicted route arcs, "
               f"{res['n_points_without_candidates']} points without candidates. Model: {res['model']}. "
               "Accuracy is computed by the API from its internal reference data.")

    gps, route = res["gps_points"], res["predicted_route"]
    fmap = folium.Map(tiles="OpenStreetMap")
    fg_gps = folium.FeatureGroup(name="GPS observations")
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=0.8).add_to(fg_gps)
    fg_gps.add_to(fmap)
    fg_route = folium.FeatureGroup(name="Predicted route")
    if route:
        folium.PolyLine(route, color="#d62728", weight=4, opacity=0.9).add_to(fg_route)
    fg_route.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    lats = [p[0] for p in gps + route]
    lons = [p[1] for p in gps + route]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st.markdown("Blue dots: GPS observations. Red line: predicted route.")
    components.html(fmap.get_root().render(), height=620)

    # offline backup: the interactive map needs internet (Leaflet JS and tiles come from a CDN)
    st.markdown("**Static view** (works without internet)")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(lons[:len(gps)], lats[:len(gps)], s=4, color="#1f77b4", label="GPS observations")
    if route:
        ax.plot([p[1] for p in route], [p[0] for p in route], color="#d62728", lw=1.8, label="Predicted route")
    ax.set_aspect(1 / max(0.2, math.cos(math.radians(sum(lats) / len(lats)))))  # equal metres per axis unit
    ax.set(xlabel="longitude", ylabel="latitude")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Timings and full metrics"):
        st.json({"metrics": m, "timings_s": res["timings_s"]})

"""Small Folium helpers shared by the EDA maps (Reet) and route maps (Rekha)."""
import folium
from branca.element import MacroElement, Template

from Reet.candidates import generate_candidates


def nearby_arcs(trip, radius=80.0):
    """Arc ids within `radius` metres of any GPS point (keeps maps light)."""
    c = generate_candidates(trip, radius=radius, k=50)
    return c.arc_id.unique()


def arc_latlon(trip, arc_id):
    a, b = trip.arcs[arc_id]
    return [(trip.node_lonlat[a, 1], trip.node_lonlat[a, 0]), (trip.node_lonlat[b, 1], trip.node_lonlat[b, 0])]


def arcs_layer(trip, arc_ids, name, color, weight=2, opacity=0.6, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    for a in arc_ids:
        folium.PolyLine(arc_latlon(trip, int(a)), color=color, weight=weight, opacity=opacity).add_to(fg)
    return fg


def route_polyline(trip, arc_seq):
    """Lat/lon polyline following a sequence of arcs (consecutive duplicates removed)."""
    pts = []
    for a in arc_seq:
        seg = arc_latlon(trip, int(a))
        if not pts or pts[-1] != seg[0]:
            pts.append(seg[0])
        pts.append(seg[1])
    return pts


def gps_layer(trip, name="GPS observations", color="#1f77b4", every=1, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    tr = trip.track.iloc[::every]
    for lat, lon, t in zip(tr.lat, tr.lon, tr.t):
        folium.CircleMarker((lat, lon), radius=2, color=color, fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(fg)
    return fg


def add_legend(m, items, title="Legend"):
    rows = "".join(
        f"<div><span style='display:inline-block;width:14px;height:4px;background:{c};"
        f"margin-right:6px;vertical-align:middle'></span>{label}</div>" for label, c in items)
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
         padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>{title}</b>{rows}
    </div>
    {{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    m.get_root().add_child(el)


def base_map(trip):
    lat0, lon0 = trip.track.lat.mean(), trip.track.lon.mean()
    m = folium.Map(location=(lat0, lon0), zoom_start=14, tiles="OpenStreetMap")
    sw = [trip.track.lat.min(), trip.track.lon.min()]
    ne = [trip.track.lat.max(), trip.track.lon.max()]
    m.fit_bounds([sw, ne])
    return m

"""Exploratory data analysis of the raw dataset (training + student-testing partitions).

Outputs go to Reet/results/:
  dataset_inventory.csv      per-trip counts, gaps, extent, network size, connectivity, file sizes
  partition_summary.csv      totals per partition
  missing_values.csv         NaN / invalid counts per file type
  label_quality.csv          per-trip distance of GPS points to their labelled route arc
  plots/*.png                distributions, trajectory + network plot, feature plots
  maps/eda_trip_<id>.html    Folium map: GPS points, road network, reference route

Run AFTER build_dataset: python -m Reet.eda
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config import LABEL_MAX_DIST_M, REET_OUT, REET_RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from Reet.data_loader import list_trips, load_trip
from Reet.features import FEATURE_COLUMNS
from Reet import folium_utils as fu

PLOTS = REET_RESULTS / "plots"
MAPS = REET_RESULTS / "maps"
MAP_TRIPS = ["00000010", "00000047"]  # training trips used for EDA maps


def raw_missing(folder):
    tid = folder.name
    out = {}
    for ext, ncol in [("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)]:
        p = folder / f"{tid}.{ext}"
        if not p.exists():
            out[f"{ext}_missing_file"] = 1
            continue
        df = pd.read_csv(p, sep=r"\s+", header=None)
        out[f"{ext}_rows"] = len(df)
        out[f"{ext}_cols"] = df.shape[1]
        out[f"{ext}_nan"] = int(df.isna().sum().sum())
        out[f"{ext}_kb"] = round(p.stat().st_size / 1024, 1)
    return out


def trip_inventory(folder, partition):
    trip = load_trip(folder)
    t = trip.track.t.to_numpy()
    dt = np.diff(t)
    x, y = trip.track.x.to_numpy(), trip.track.y.to_numpy()
    n_nodes = len(trip.node_xy)
    a = trip.arcs
    g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    n_weak, lab_weak = connected_components(g, directed=True, connection="weak")
    n_strong, lab_strong = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(a.ravel(), minlength=n_nodes)
    route = trip.route
    route_connected = float(np.mean(a[route[:-1], 1] == a[route[1:], 0])) if len(route) > 1 else np.nan
    row = dict(
        partition=partition, trip_id=trip.trip_id,
        points=trip.n_points, duration_s=float(t[-1] - t[0]),
        dt_median=float(np.median(dt)), dt_max=float(dt.max()), share_dt_1s=float(np.mean(dt == 1)),
        gaps_over_10s=int((dt > 10).sum()), gaps_over_60s=int((dt > 60).sum()),
        extent_x_km=float((x.max() - x.min()) / 1000), extent_y_km=float((y.max() - y.min()) / 1000),
        track_length_km=float(np.hypot(np.diff(x), np.diff(y)).sum() / 1000),
        lon_min=trip.track.lon.min(), lon_max=trip.track.lon.max(),
        lat_min=trip.track.lat.min(), lat_max=trip.track.lat.max(),
        nodes=n_nodes, arcs=len(a), isolated_nodes=int((deg == 0).sum()),
        weak_components=int(n_weak), largest_weak_share=float(np.bincount(lab_weak).max() / n_nodes),
        strong_components=int(n_strong),
        arc_len_median_m=float(np.median(trip.arc_len)), arc_len_max_m=float(trip.arc_len.max()),
        route_arcs=len(route), route_length_km=float(trip.arc_len[route].sum() / 1000),
        route_consecutive_connected=route_connected,
        **{k: v for k, v in trip.cleaning.items()},
    )
    row.update(raw_missing(folder))
    return row


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)


def plots_inventory(inv):
    tr = inv[inv.partition == "train"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].hist(tr.points, bins=30, color="#4C72B0")
    axs[0].set(title="GPS points per trip (train)", xlabel="points", ylabel="trips")
    axs[1].hist(tr.track_length_km, bins=30, color="#55A868")
    axs[1].set(title="Track length per trip (train)", xlabel="km")
    axs[2].hist(np.log10(tr.arcs), bins=30, color="#C44E52")
    axs[2].set(title="Road-network size (train)", xlabel="log10(arcs)")
    save(fig, "trip_size_distributions.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(inv.trip_id, inv.gaps_over_10s, color=np.where(inv.partition == "train", "#4C72B0", "#DD8452"))
    ax.set(title="Number of time gaps > 10 s per trip (blue train, orange student test)",
           xlabel="trip", ylabel="gaps")
    ax.set_xticks([])
    save(fig, "time_gaps_per_trip.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for part, c in [("train", "#4C72B0"), ("student_test", "#DD8452")]:
        s = inv[inv.partition == part]
        ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=25, color=c, label=part)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set(title="Trip locations (centre of each trajectory): trips span several continents",
           xlabel="longitude", ylabel="latitude")
    save(fig, "trip_locations_world.png")


def plots_time_gaps():
    dts = []
    for folder in list_trips(TRAIN_DIR):
        t = np.loadtxt(folder / f"{folder.name}.track", usecols=2)
        dts.append(np.diff(np.sort(t)))
    dt = np.concatenate(dts)
    vals, counts = np.unique(np.clip(dt, 0, 20).round(), return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vals, counts, color="#4C72B0")
    ax.set_yscale("log")
    ax.set(title="Sampling interval between GPS fixes (train, clipped at 20 s)",
           xlabel="seconds", ylabel="count (log)")
    save(fig, "sampling_intervals.png")
    return pd.Series(dt).describe()


def plot_trip_network(trip_id):
    trip = load_trip(TRAIN_DIR / trip_id)
    arcs = fu.nearby_arcs(trip, radius=300)
    fig, ax = plt.subplots(figsize=(7, 7))
    for a in arcs:
        ax.plot([trip.ax[a], trip.bx[a]], [trip.ay[a], trip.by[a]], color="#bbbbbb", lw=0.6)
    r = trip.route
    for a in r:
        ax.plot([trip.ax[a], trip.bx[a]], [trip.ay[a], trip.by[a]], color="#2ca02c", lw=2.5)
    ax.scatter(trip.track.x, trip.track.y, s=2, color="#1f77b4", zorder=3)
    ax.set_aspect("equal")
    ax.set(title=f"Trip {trip_id} (train): GPS (blue), route (green), network (grey)",
           xlabel=f"x (m, {trip.crs})", ylabel="y (m)")
    save(fig, f"trip_{trip_id}_network.png")


def folium_map(trip_id):
    trip = load_trip(TRAIN_DIR / trip_id)
    m = fu.base_map(trip)
    fu.arcs_layer(trip, fu.nearby_arcs(trip, 80), "Road network (within 80 m)", "#888888", weight=1.5,
                  opacity=0.5).add_to(m)
    folium.PolyLine(fu.route_polyline(trip, trip.route), color="#2ca02c", weight=5, opacity=0.8,
                    tooltip="Reference route").add_to(folium.FeatureGroup(name="Reference route").add_to(m))
    fu.gps_layer(trip).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    fu.add_legend(m, [("GPS observations", "#1f77b4"), ("Reference route", "#2ca02c"),
                      ("Road network", "#888888")], title=f"Trip {trip_id} (train)")
    m.save(str(MAPS / f"eda_trip_{trip_id}.html"))


def plots_labels_and_features():
    pts = pd.read_parquet(REET_OUT / "train_points.parquet")
    q = pts.groupby("trip_id").agg(points=("point_idx", "size"),
                                   label_dist_median=("label_dist", "median"),
                                   label_dist_p95=("label_dist", lambda s: s.quantile(0.95)),
                                   unreliable_share=("reliable", lambda s: 1 - s.mean()),
                                   miss_rate_reliable=("has_true_cand", "mean"))
    q["miss_rate_reliable"] = 1 - pts[pts.reliable].groupby("trip_id").has_true_cand.mean()
    q.sort_values("unreliable_share", ascending=False).to_csv(REET_RESULTS / "label_quality.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(pts.label_dist, 0, 150), bins=75, color="#4C72B0")
    ax.axvline(LABEL_MAX_DIST_M, color="red", ls="--", label=f"reliability threshold {LABEL_MAX_DIST_M:.0f} m")
    ax.set_yscale("log")
    ax.set(title="Distance from GPS point to its labelled route arc (train)", xlabel="metres (clipped 150)",
           ylabel="points (log)")
    ax.legend()
    save(fig, "label_distance.png")

    c = pd.read_parquet(REET_OUT / "train_candidates.parquet")
    c = c[c.reliable]
    show = ["dist", "heading_diff", "win_dist_mean", "dist_minus_min", "rank", "speed"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axs.ravel(), show):
        for lab, color, name in [(0, "#C44E52", "wrong candidate"), (1, "#55A868", "true arc")]:
            v = c.loc[c.label == lab, col]
            ax.hist(v.clip(upper=v.quantile(0.99)), bins=40, alpha=0.55, density=True, color=color, label=name)
        ax.set_title(col)
    axs[0, 0].legend()
    fig.suptitle("Feature distributions by label (train, reliable points)")
    save(fig, "feature_distributions.png")

    corr = c[FEATURE_COLUMNS + ["label"]].sample(min(len(c), 200000), random_state=0).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im)
    ax.set_title("Feature correlation (train)")
    save(fig, "feature_correlation.png")

    bal = c.label.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["wrong (0)", "true (1)"], bal.values, color=["#C44E52", "#55A868"])
    ax.set(title="Class balance of candidate rows (train)", ylabel="rows")
    save(fig, "class_balance.png")
    return q


def main():
    for d in (REET_RESULTS, PLOTS, MAPS):
        d.mkdir(parents=True, exist_ok=True)
    rows = [trip_inventory(f, "train") for f in list_trips(TRAIN_DIR)]
    rows += [trip_inventory(f, "student_test") for f in list_trips(STUDENT_TEST_DIR)]
    inv = pd.DataFrame(rows)
    inv.to_csv(REET_RESULTS / "dataset_inventory.csv", index=False)

    summary = inv.groupby("partition").agg(
        trips=("trip_id", "size"), points=("points", "sum"), points_median=("points", "median"),
        points_min=("points", "min"), points_max=("points", "max"),
        nodes_median=("nodes", "median"), arcs_median=("arcs", "median"),
        route_arcs_median=("route_arcs", "median"), trips_with_gap_over_60s=("gaps_over_60s", lambda s: (s > 0).sum()),
        isolated_nodes_total=("isolated_nodes", "sum"), weak_components_median=("weak_components", "median"),
        route_connected_mean=("route_consecutive_connected", "mean"))
    summary.to_csv(REET_RESULTS / "partition_summary.csv")
    print(summary.T)

    miss_cols = [c for c in inv.columns if c.endswith("_nan") or c.endswith("_missing_file")
                 or c in ("invalid_dropped", "duplicate_time_dropped", "speed_spikes_dropped")]
    inv.groupby("partition")[miss_cols].sum().T.to_csv(REET_RESULTS / "missing_values.csv")

    plots_inventory(inv)
    print(plots_time_gaps())
    for t in MAP_TRIPS:
        plot_trip_network(t)
        folium_map(t)
    q = plots_labels_and_features()
    print(q.sort_values("unreliable_share", ascending=False).head(5))


if __name__ == "__main__":
    main()

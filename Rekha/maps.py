"""Folium maps comparing GPS observations, reference route and predicted routes.

Uses predictions saved by evaluate_hmm.py. One interactive HTML per trip with
layer control + legend, plus a static PNG for the report.

Run: python -m Rekha.maps
"""
import numpy as np
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import REKHA_RESULTS, STUDENT_TEST_DIR
from Reet import folium_utils as fu
from Reet.data_loader import list_trips, load_trip
from Rekha.hmm import route_sequence

MAPS = REKHA_RESULTS / "maps"
COLORS = {"gps": "#1f77b4", "ref": "#2ca02c", "hmm": "#d62728", "clf": "#ff7f0e", "net": "#999999"}


def trip_map(trip, preds):
    clf, hmm_d, hmm = preds
    m = fu.base_map(trip)
    fu.arcs_layer(trip, fu.nearby_arcs(trip, 60), "Road network (within 60 m)", COLORS["net"],
                  weight=1, opacity=0.5, show=False).add_to(m)
    fg = folium.FeatureGroup(name="Reference route (server-side ground truth)")
    folium.PolyLine(fu.route_polyline(trip, trip.route), color=COLORS["ref"], weight=7, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fu.arcs_layer(trip, np.unique(clf[clf >= 0]), "Classifier-only arcs", COLORS["clf"], weight=3,
                  opacity=0.8, show=False).add_to(m)
    fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
    folium.PolyLine(fu.route_polyline(trip, route_sequence(hmm)), color=COLORS["hmm"], weight=3,
                    opacity=0.9).add_to(fg)
    fg.add_to(m)
    fu.gps_layer(trip, color=COLORS["gps"]).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    fu.add_legend(m, [("GPS observations", COLORS["gps"]), ("Reference route", COLORS["ref"]),
                      ("Predicted route (HMM)", COLORS["hmm"]), ("Classifier-only arcs", COLORS["clf"]),
                      ("Road network", COLORS["net"])], title=f"Trip {trip.trip_id} (student test)")
    m.save(str(MAPS / f"route_trip_{trip.trip_id}.html"))


def static_plot(trip, preds):
    clf, _, hmm = preds
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, pred, title in [(axs[0], clf, "Classifier-only"), (axs[1], hmm, "Classifier + HMM")]:
        for a in np.unique(trip.route):
            ax.plot([trip.ax[a], trip.bx[a]], [trip.ay[a], trip.by[a]], color=COLORS["ref"], lw=5, alpha=0.4)
        for a in np.unique(pred[pred >= 0]):
            ax.plot([trip.ax[a], trip.bx[a]], [trip.ay[a], trip.by[a]], color=COLORS["hmm"], lw=1.5)
        ax.scatter(trip.track.x, trip.track.y, s=1.5, color=COLORS["gps"])
        ax.set_title(f"{title}: red = predicted arcs, green = reference, blue = GPS")
        ax.set_aspect("equal")
    fig.suptitle(f"Trip {trip.trip_id} (student test)")
    fig.tight_layout()
    fig.savefig(MAPS / f"route_trip_{trip.trip_id}.png", dpi=130)
    plt.close(fig)


def main():
    MAPS.mkdir(parents=True, exist_ok=True)
    for folder in list_trips(STUDENT_TEST_DIR):
        f = REKHA_RESULTS / f"pred_{folder.name}.npy"
        if not f.exists():
            continue
        trip = load_trip(folder)
        preds = np.load(f)
        trip_map(trip, preds)
        static_plot(trip, preds)
        print("map saved for", trip.trip_id)


if __name__ == "__main__":
    main()

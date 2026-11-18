# Reet – Data, EDA and features

## Files

| File | Purpose |
| --- | --- |
| `data_loader.py` | `load_trip(folder, with_route)` reads `.track/.nodes/.arcs/.route`, cleans the track (invalid rows, duplicate timestamps, speed spikes > 70 m/s), projects each trip to its own UTM zone, computes arc length/bearing, crops the network to the track area (+600 m) |
| `geometry.py` | per-trip UTM zone selection and projection (pyproj), point-to-segment distance, bearings |
| `labels.py` | per-point labels from the ordered route by monotone dynamic-programming alignment |
| `candidates.py` | candidate arcs per GPS point (KD-tree over arc samples, radius 50 m, max 8) |
| `features.py` | 15 features per (point, candidate) pair, `FEATURE_COLUMNS` |
| `candidate_study.py` | candidate miss rate for radius × k on the training partition |
| `build_dataset.py` | builds `output/*_candidates.parquet` and `output/*_points.parquet` |
| `eda.py` | inventory, missing values, connectivity, time gaps, plots, Folium maps |
| `folium_utils.py` | map helpers shared with Rekha |

## Run

```bash
python -m Reet.candidate_study
python -m Reet.build_dataset
python -m Reet.eda
```

## Outputs

- `results/dataset_inventory.csv` per-trip points, duration, time gaps, extent, nodes, arcs, isolated nodes, connected components, route length, file sizes, NaN counts
- `results/partition_summary.csv`, `results/missing_values.csv`, `results/label_quality.csv`
- `results/candidate_miss_rate.csv` + `.png`
- `results/plots/` trip sizes, sampling intervals, gaps, extents, trajectory + network, label distance, feature distributions, correlation, class balance
- `results/maps/eda_trip_*.html` Folium maps (GPS, network, reference route, legend, layer control)
- `output/` parquet datasets used by Manasa (rebuild with `build_dataset`, not committed)

## Key decisions (be ready to explain)

1. **Metres, not degrees, and one UTM zone per trip.** The trips are on several continents (North and South America, Europe, Russia, ...; see `trip_locations_world.png`). One degree of longitude is 111 km at the equator but only about 62 km at 56 N, so degree distances are distorted. Each trip is projected to the UTM zone of its centre (EPSG:326xx north, 327xx south); a single fixed zone would distort trips far from its central meridian. The zone is stored in `dataset_inventory.csv` (`crs`).
2. **Labels.** The route lists arcs in order but has no per-point label. Each point is assigned to a route position with a monotone alignment: `cost(i, j) = dist(i, route[j]) + min_{k <= j} cost(i-1, k)`, so the matched position never goes backwards in time.
3. **Unreliable labels.** Points more than 50 m from their labelled route arc are flagged `reliable = False`. Trips 00000054 and 00000033 have most points far from their route (the reference route covers only part of the track). These points are excluded from training and scoring.
4. **Candidates.** Radius 50 m and k = 8 give a miss rate below 1% on reliable training points with about 6 candidates per point (see `candidate_miss_rate.csv`).
5. **Features.** Distance and rank say how close a candidate is; heading difference separates the two directions of a two-way road (they share the same geometry); window distance uses neighbouring points to resist noise; degree features describe junction complexity.

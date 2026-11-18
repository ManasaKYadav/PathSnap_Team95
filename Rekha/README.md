# Rekha – HMM, Viterbi and route evaluation

## Files

| File | Purpose |
| --- | --- |
| `road_graph.py` | directed road graph (scipy sparse), Dijkstra shortest paths with a distance limit, cached |
| `hmm.py` | emission and transition models, Viterbi decoder, classifier-only baseline, route sequence |
| `route_metrics.py` | pointwise accuracy, route continuity, length-weighted route precision / recall / F1 |
| `pipeline.py` | `run_pipeline(trip_folder)`: load (no route) → features → classifier → Viterbi |
| `tune_decoder.py` | grid search of β, emission weight, σ on 10 held-out training trips |
| `evaluate_hmm.py` | classifier-only vs distance-HMM vs classifier-HMM on the 10 student-test trips |
| `maps.py` | Folium route maps (HTML) and static PNGs |
| `decoder_config.json` | tuned decoder parameters used by the pipeline and the API |

## Run

```bash
python -m Rekha.tune_decoder
python -m Rekha.evaluate_hmm
python -m Rekha.maps
```

## HMM design (be ready to explain)

- **Hidden states:** for GPS point i, the states are its candidate arcs (from Reet). Each state keeps its arc id, so the decoded state maps straight back to a road segment.
- **Emission:** `w · log(p / Σp)`, where p is the classifier probability of the candidate, normalised over the candidates of the same point. Baseline without the classifier: `−½ (dist / σ)²`.
- **Transition:** `−|d_route − d_gps| / (β · √Δt)`.
    - `d_gps` is the straight-line distance between the two GPS points.
    - `d_route` is the network distance between their projections: on the same arc it is `s_b − s_a`; otherwise it is the rest of arc a, plus the shortest path from the end of a to the start of b, plus `s_b`.
    - A real vehicle drives a network distance close to the straight-line distance, so large differences are penalised. Longer time gaps (Δt) tolerate more difference.
- **Invalid transitions:** if there is no path within the search limit, or the move goes backwards along an arc (more than 5 m), the transition gets the floor value `−max_penalty`. The same floor applies to very poor transitions. The decoder can still jump across a disconnected part of the network, but at a high cost, so it never breaks.
- **Missing candidates:** points with no candidate within 50 m have no state. They are skipped by the decoder and forward-filled.
- **GPS gaps:** steps longer than 300 m use a larger shortest-path search limit (3 × gap + 500 m). The √Δt factor relaxes the transition for long time gaps.
- **Viterbi (log space):** `δ_i(b) = e_i(b) + max_a [δ_{i−1}(a) + T(a, b)]`, with back-pointers `ψ_i(b) = argmax_a`. Backtracking from the best final state gives one arc per point. Log space turns products of small probabilities into sums, which avoids numerical underflow.
- **Decoder tuning without leakage:** the classifier is refitted on 70 training trips and the decoder grid is scored on the other 10 training trips. The student-test trips are used only for the final comparison.

## Outputs

- `results/decoder_grid.csv` all grid results
- `results/hmm_comparison_per_trip.csv`, `results/hmm_comparison_summary.csv`, `results/hmm_changes.csv` (points the HMM fixed or broke)
- `results/plots/` hmm_vs_classifier, pointwise_accuracy_per_trip, decoder_grid
- `results/maps/route_trip_*.html` interactive maps; `route_trip_*.png` static versions

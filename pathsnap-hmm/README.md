# HMM and Viterbi route reconstruction

- `hmm.py`: road graph, Dijkstra distances, Viterbi decoder, route stitching and route overlap.
- `decode.py`: tunes the decoder on 24 training trips using out-of-fold probabilities from
  `pathsnap-tuning`, then compares classifier-only against classifier + HMM on student_test.

**States:** the 8 candidate arcs of each GPS reading, from `pathsnap-training/data.py`.

**Emission:** `w_emit · log p`, where `p` is the tuned classifier score normalised over the 8 candidates.

**Transition:** `-|route distance − straight distance| / β`. The route distance is the
shortest network path between the two projections (Dijkstra). β grows with the time gap:
`β + β_gap · (Δt − 1)`. Unreachable pairs get a fixed penalty (`disconnect_logp`) instead of
zero probability, so Viterbi never runs out of paths. These are counted as breaks.

**Stitching:** consecutive decoded arcs that don't touch are joined by a shortest path,
which gives a drivable route.

```bash
cd pathsnap-hmm
python decode.py --cache ../pathsnap-training/cache
```

Outputs in `results/`:

| file | contents |
|---|---|
| `decoder_grid_train.csv`, `decoder_tuning.json` | the grid search and the chosen settings |
| `decoder_params.json` | the settings the API uses |
| `hmm_vs_classifier_student_test.csv`, `hmm_summary_student_test.json` | per-trip and overall comparison |
| `student_test_*_predictions.npz`, `student_test_hmm_routes.npz` | predictions, used for the maps |

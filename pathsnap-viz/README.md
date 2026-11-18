# Maps and figures

Run after `pathsnap-eda`, `pathsnap-tuning` and `pathsnap-hmm`.

```bash
cd pathsnap-viz
python make_maps.py --cache ../pathsnap-training/cache   # Folium maps  -> maps/
python plots.py --cache ../pathsnap-training/cache       # all figures  -> figures/
```

**Maps.** `maps/prediction_<trip>.html` has one map per student_test trip, with
toggleable layers: road network, reference route, classifier-only route, HMM route,
GPS readings, and start/end markers. `maps/eda_*.html` are the EDA maps and the
map of trip locations. Open them in a browser; the background needs internet.

**Figures.** `figures/` holds:
- the model comparison (CV with fold dots, train vs CV vs test, timings);
- HistGB boosting curves;
- the Optuna history and tuned vs untuned comparison;
- the decoder grid and HMM vs classifier comparison;
- the error-by-speed analysis.

The CSV/JSON files behind them are saved next to the figures.

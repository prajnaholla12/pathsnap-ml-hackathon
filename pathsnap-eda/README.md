# Dataset exploration

`eda.py` reads the feature cache from `pathsnap-training/data.py`. It uses **train and
student_test only**.

```bash
cd pathsnap-eda
python eda.py --cache ../pathsnap-training/cache
```

Outputs in `results/`:

| file | contents |
|---|---|
| `eda_inventory.csv` | one row per trip: size, timestamp issues, gaps, network connectivity, two-way share, reference alignment, candidate miss rate |
| `eda_summary.csv` | totals per partition |
| `projection_check.csv` | local projection vs haversine |
| `eda_*.png` | the figures used in report section 2 |

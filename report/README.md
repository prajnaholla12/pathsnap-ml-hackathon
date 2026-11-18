# Technical report

`PathSnap_Report.pdf` is generated from the outputs of every stage, so its numbers
match the repository.

```bash
pip install playwright && playwright install chromium     # Python 3.12+
python report/screenshot_maps.py pathsnap-viz/maps/prediction_00000056.html report/img/map_pred_56.png
python report/build_report.py
```

`img/` holds the UI and map screenshots used in the report. Before submitting, add
the team names/SRNs on the title page and replace the API-log section with a
screenshot of the terminal on Laptop 1 from the real evaluation run.

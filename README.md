# PathSnap – GPS map-matching and route reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR campus).

For every GPS reading of a trip, PathSnap predicts the road arc the vehicle was on and
rebuilds the driven route. Each stage has its own folder and README:

| stage | folder | what it does |
|---|---|---|
| 1 | [pathsnap-training](pathsnap-training/) | candidates, features, label alignment, five baseline classifiers |
| 2 | [pathsnap-eda](pathsnap-eda/) | dataset exploration |
| 3 | [pathsnap-tuning](pathsnap-tuning/) | Bayesian (Optuna TPE) tuning of HistGradientBoosting |
| 4 | [pathsnap-hmm](pathsnap-hmm/) | HMM + Viterbi decoding, decoder tuning, comparison |
| 5 | [pathsnap-viz](pathsnap-viz/) | Folium maps and all figures |
| 6 | [pathsnap-api](pathsnap-api/) | FastAPI backend (Laptop 1) |
| 7 | [pathsnap-ui](pathsnap-ui/) | evaluator page (Laptop 2) |
| – | [report](report/) | technical report (PDF) |

## Results on student_test (10 trips, 19,160 readings)

| system | top-1 accuracy | Macro F1 | route F1 |
|---|---|---|---|
| HistGB (baseline, train.py) | 85.30% | 0.7571 | – |
| HistGB tuned | 85.81% | 0.7700 | 0.806 |
| HistGB tuned + HMM/Viterbi | **91.75%** | **0.9033** | **0.972** |

## Reproduce everything

```bash
cd pathsnap-training
python -m pip install -r requirements.txt
python data.py --zip "<path>/Student folder.zip" --out cache
python train.py --cache cache --out results
cd ../pathsnap-eda    && python eda.py --cache ../pathsnap-training/cache
cd ../pathsnap-tuning && python -m pip install -r requirements.txt && python tune.py --cache ../pathsnap-training/cache --trials 20
cd ../pathsnap-hmm    && python decode.py --cache ../pathsnap-training/cache
cd ../pathsnap-viz    && python -m pip install -r requirements.txt && python make_maps.py --cache ../pathsnap-training/cache && python plots.py --cache ../pathsnap-training/cache
cd .. && python report/build_report.py
```

## Final evaluation (two laptops)

**Laptop 1 (API):** copy the ten evaluation trips to `data/eval/<trip_id>/`, then run:

```bash
cd pathsnap-api && python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Keep this terminal open; it is the API log the evaluator sees.

**Laptop 2 (UI):**

```bash
cd pathsnap-ui && python -m http.server 5500
```

Open `http://localhost:5500`, enter `http://<laptop-1-ip>:8000` and press Connect.
The ten trips appear. Selecting one sends only its id to the API, which runs the
whole pipeline and scores it on the server. The reference route never leaves Laptop 1.

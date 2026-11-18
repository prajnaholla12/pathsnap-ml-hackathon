# PathSnap API (Laptop 1)

FastAPI backend. For a trip id it runs the full pipeline:

1. `data.prepare_candidates`: candidates and features.
2. The tuned classifier.
3. HMM/Viterbi decoding and stitching.

It then scores the result on the server against the trip's reference route, using the same
label alignment and metrics as `pathsnap-training`.

```bash
cd pathsnap-api
python -m pip install -r requirements.txt
# put the ten evaluation trips in ../data/eval/<trip_id>/<trip_id>.{track,nodes,arcs,route}
uvicorn main:app --host 0.0.0.0 --port 8000
```

Find the laptop's IP with `ipconfig` (Windows) or `ip a`, and allow port 8000 through
the firewall. Keep this terminal visible during the evaluation: every request is logged
with the client IP, trip id, timings and accuracy.

| method | path | body / response |
|---|---|---|
| GET | `/health` | status, number of trips, model file |
| GET | `/trips` | `{"trips": [{"trip_id", "n_points"}]}` |
| POST | `/predict` | `{"trip_id": "..."}` → GPS, HMM route, classifier-only route, per-reading arcs, candidate network, metrics, timings |

Environment variables: `PATHSNAP_EVAL_DIR` (default `../data/eval`), `PATHSNAP_MODEL`, `PATHSNAP_DECODER`.

**Ground truth stays on the server.** `load_trip_inputs` reads only `.track`, `.nodes` and
`.arcs`. The `.route` file is opened only in `scoring.py`, after prediction, and only the
metric values are returned. No route geometry is sent to the UI.
Missing reference files leave prediction available with JSON `null` accuracy
scores. Invalid trip input files or invalid references return HTTP 422. Unknown
trip IDs return HTTP 404. Metrics also expose `unfilled_gaps`, because stitching
cannot guarantee connectivity when no connecting path is found.

## Verify the API before committing

From the repository root, using the same virtual environment as the server:

```powershell
python -m pip install -r pathsnap-api/requirements-test.txt
python pathsnap-api/smoke_test.py --student-zip "$env:USERPROFILE\Downloads\Student folder.zip"
```

The check uses temporary copies of the ten student-test trips. It runs real model
inference, checks the response fields used by the UI, compares predictions with
the supplied HMM outputs, and checks unknown IDs, invalid inputs, and prediction
without a reference. It never opens reserved evaluator data or changes the
supplied experiment outputs. No separate server process is needed for this check.

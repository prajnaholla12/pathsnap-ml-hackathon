"""PathSnap prediction API (Laptop 1).

    uvicorn main:app --host 0.0.0.0 --port 8000        (run from this folder)

Environment variables:
    PATHSNAP_EVAL_DIR  folder with one sub-folder per evaluation trip:
                       <id>/<id>.track, .nodes, .arcs, .route   (default ../data/eval)
    PATHSNAP_MODEL     classifier (default ../pathsnap-tuning/results/tuned_model/model.joblib)
    PATHSNAP_DECODER   decoder settings (default ../pathsnap-hmm/results/decoder_params.json)

The prediction path (load_trip_inputs -> prepare_candidates -> classifier -> Viterbi)
never opens the .route file. The route is read afterwards in scoring.py, only to
compute accuracy, and its geometry is never returned to the UI.
"""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pathsnap-training"))
sys.path.insert(0, str(ROOT / "pathsnap-hmm"))
from data import prepare_candidates  # noqa: E402
from train import FEATURES  # noqa: E402
from hmm import DecoderParams, RoadGraph, stitch, viterbi  # noqa: E402
from scoring import score_trip  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "eval"))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "pathsnap-tuning" / "results" / "tuned_model" / "model.joblib"))
DECODER_PATH = Path(os.environ.get("PATHSNAP_DECODER", ROOT / "pathsnap-hmm" / "results" / "decoder_params.json"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")
STATE = {}


def short(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


@asynccontextmanager
async def lifespan(_app):
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["decoder"] = DecoderParams(**json.loads(DECODER_PATH.read_text())) if DECODER_PATH.exists() else DecoderParams()
    STATE["trips"] = sorted(p.name for p in EVAL_DIR.iterdir() if (p / f"{p.name}.track").exists()) if EVAL_DIR.is_dir() else []
    log.info("model: %s", short(MODEL_PATH))
    log.info("decoder: %s", STATE["decoder"])
    log.info("%d evaluation trips in %s: %s", len(STATE["trips"]), short(EVAL_DIR), STATE["trips"])
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def load_trip_inputs(trip_id):
    """GPS + road graph only. Same cleaning as pathsnap-training/data.py."""
    base = EVAL_DIR / trip_id / trip_id
    track = np.loadtxt(f"{base}.track", dtype=float, ndmin=2)
    nodes = np.loadtxt(f"{base}.nodes", dtype=float, ndmin=2)
    arcs = np.loadtxt(f"{base}.arcs", dtype=int, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Expected 3 track columns, 2 node columns and 2 arc columns")
    if not len(nodes) or not len(arcs) or not np.isfinite(nodes).all():
        raise ValueError("Road network must contain finite nodes and at least one arc")
    if np.any((arcs < 0) | (arcs >= len(nodes))):
        raise ValueError("Road network contains an out-of-range node ID")
    valid = np.isfinite(track).all(axis=1) & (abs(track[:, 0]) <= 180) & (abs(track[:, 1]) <= 90)
    track = track[valid]
    if not len(track):
        raise ValueError("Trip contains no valid GPS observations")
    track = track[np.argsort(track[:, 2], kind="stable")]
    return track, nodes, arcs


def run_pipeline(track, nodes, arcs):
    t0 = perf_counter()
    points, starts, vectors, candidates, features = prepare_candidates(track, nodes, arcs)
    t1 = perf_counter()
    raw = STATE["model"].predict_proba(features.reshape(-1, len(FEATURES)))[:, 1].reshape(candidates.shape)
    prob = np.maximum(raw, 1e-12)
    prob /= prob.sum(axis=1, keepdims=True)
    t2 = perf_counter()
    graph = RoadGraph(arcs, starts, vectors)
    hmm_arcs, breaks = viterbi(points, track[:, 2], candidates, prob, graph, STATE["decoder"])
    route, unfilled = stitch(graph, hmm_arcs)
    t3 = perf_counter()
    clf_arcs = candidates[np.arange(len(prob)), np.argmax(prob, axis=1)]
    return {"points": points, "starts": starts, "vectors": vectors, "candidates": candidates, "prob": prob,
            "graph": graph, "hmm_arcs": hmm_arcs, "clf_arcs": clf_arcs, "route": route,
            "breaks": breaks, "unfilled": unfilled,
            "timing_s": {"candidates_features": round(t1 - t0, 3), "classifier": round(t2 - t1, 3),
                         "viterbi": round(t3 - t2, 3)}}


def lines(nodes, arcs, sequence, connect=True):
    out, current, last = [], [], None
    for a in sequence:
        f, t = arcs[a]
        if last is None or f != last or not connect:
            if current:
                out.append(current)
            current = [[round(nodes[f, 1], 6), round(nodes[f, 0], 6)]]
        current.append([round(nodes[t, 1], 6), round(nodes[t, 0], 6)])
        last = t
    if current:
        out.append(current)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "trips": len(STATE["trips"]), "model": MODEL_PATH.name}


@app.get("/trips")
def trips(request: Request):
    log.info("GET /trips from %s", request.client.host if request.client else "?")
    listing = []
    for t in STATE["trips"]:
        with open(EVAL_DIR / t / f"{t}.track") as f:
            listing.append({"trip_id": t, "n_points": sum(1 for _ in f)})
    return {"trips": listing}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    trip_id = req.trip_id.strip()
    log.info(">>> POST /predict trip=%s from %s", trip_id, request.client.host if request.client else "?")
    if trip_id not in STATE["trips"]:
        log.warning("unknown trip id %s", trip_id)
        raise HTTPException(404, f"trip {trip_id} is not in the evaluation set")
    start = perf_counter()
    try:
        track, nodes, arcs = load_trip_inputs(trip_id)
    except (OSError, ValueError) as exc:
        log.warning("invalid input for trip %s: %s", trip_id, exc)
        raise HTTPException(422, "Trip input files are missing or invalid") from exc
    result = run_pipeline(track, nodes, arcs)
    log.info("    %d GPS readings, %d arcs in network | features %.2fs, classifier %.2fs, viterbi %.2fs",
             len(track), len(arcs), *result["timing_s"].values())
    try:
        metrics = score_trip(EVAL_DIR, trip_id, track, arcs, result)
    except (OSError, ValueError) as exc:
        log.warning("invalid reference for trip %s: %s", trip_id, exc)
        raise HTTPException(422, "Server-side reference file is invalid") from exc
    total = perf_counter() - start
    if metrics["has_reference"]:
        log.info("    accuracy: HMM top-1 %.2f%% | classifier-only %.2f%% | HMM macro F1 %.4f | route F1 %.3f | %.2fs total",
                 100 * metrics["hmm_top1_accuracy"], 100 * metrics["classifier_top1_accuracy"],
                 metrics["hmm_macro_f1"], metrics["route_f1"], total)
    else:
        log.info("    no reference: prediction returned without accuracy scores | %.2fs total", total)
    shown = np.unique(result["candidates"])[:6000]
    runs = result["clf_arcs"][np.r_[True, result["clf_arcs"][1:] != result["clf_arcs"][:-1]]]
    return {
        "trip_id": trip_id,
        "n_points": int(len(track)),
        "gps": [[round(lat, 6), round(lon, 6)] for lon, lat in track[:, :2]],
        "predicted_route": lines(nodes, arcs, result["route"]),
        "classifier_route": lines(nodes, arcs, runs, connect=False),
        "predicted_arcs_per_point": result["hmm_arcs"].tolist(),
        "network": [[[round(nodes[f, 1], 6), round(nodes[f, 0], 6)], [round(nodes[t, 1], 6), round(nodes[t, 0], 6)]]
                    for f, t in arcs[shown]],
        "metrics": metrics,
        "timing_s": {**result["timing_s"], "total": round(total, 3)},
    }

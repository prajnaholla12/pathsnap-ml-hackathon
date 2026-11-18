"""Tune the HMM decoder on training trips, then compare classifier-only vs HMM on student_test.

Decoder settings are chosen on 24 training trips using out-of-fold probabilities
from pathsnap-tuning (each trip scored by the fold model that never saw it).
Selection metric: mean per-trip Macro F1, as in pathsnap-training.
"""
import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "pathsnap-training"
sys.path.insert(0, str(TRAINING))
from data import project  # noqa: E402
from train import SEED, read_trips, save_csv, score, write_json  # noqa: E402
from hmm import DecoderParams, RoadGraph, route_overlap, stitch, viterbi  # noqa: E402

GRID = list(itertools.product([0.25, 0.5, 1.0, 2.0], [2.0, 5.0, 10.0, 25.0, 50.0], [1.0, 5.0]))


def geometry(trip):
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    origin = np.mean(track[:, :2], axis=0)
    points = project(track[:, :2], origin)
    xy = project(nodes, origin)
    starts = xy[arcs[:, 0]]
    return points, starts, xy[arcs[:, 1]] - starts


def classifier_only(trip, probabilities):
    return trip["candidates"][np.arange(len(probabilities)), np.argmax(probabilities, axis=1)]


def run(cache, tuning_results, out):
    out.mkdir(parents=True, exist_ok=True)
    training, testing = read_trips(cache, "train"), read_trips(cache, "student_test")
    folder = tuning_results / "tuned_model"
    oof = dict(np.load(folder / "train_oof_probabilities.npz"))
    test_prob = dict(np.load(folder / "student_test_probabilities.npz"))

    rng = np.random.RandomState(SEED)
    pool = sorted(t for t, trip in training.items() if len(trip["labels"]) < 6000)
    tune_ids = sorted(rng.choice(pool, 24, replace=False).tolist())
    prepared = []
    for t in tune_ids:
        trip = training[t]
        points, starts, vectors = geometry(trip)
        prepared.append((t, trip, points, RoadGraph(trip["arcs"], starts, vectors)))

    grid_rows = []
    for w, beta, gap in GRID:
        params = DecoderParams(w_emit=w, beta=beta, beta_gap=gap)
        rows = []
        for t, trip, points, graph in prepared:
            graph.cache = {}
            arcs, _ = viterbi(points, trip["track"][:, 2], trip["candidates"], oof[t], graph, params)
            rows.append(score(trip, arcs, oof[t]))
        n = sum(r["observations"] for r in rows)
        grid_rows.append({"w_emit": w, "beta": beta, "beta_gap": gap,
                          "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                          "top1_accuracy": sum(r["top1_accuracy"] * r["observations"] for r in rows) / n})
        print(grid_rows[-1], flush=True)
    save_csv(out / "decoder_grid_train.csv", grid_rows)
    best = max(grid_rows, key=lambda r: r["macro_f1"])
    params = DecoderParams(w_emit=best["w_emit"], beta=best["beta"], beta_gap=best["beta_gap"])
    clf_rows = [score(trip, classifier_only(trip, oof[t]), oof[t]) for t, trip, _, _ in prepared]
    n = sum(r["observations"] for r in clf_rows)
    write_json(out / "decoder_params.json", asdict(params))
    write_json(out / "decoder_tuning.json", {
        "tuning_trips": tune_ids, "selected": best,
        "classifier_only_macro_f1": float(np.mean([r["macro_f1"] for r in clf_rows])),
        "classifier_only_top1_accuracy": sum(r["top1_accuracy"] * r["observations"] for r in clf_rows) / n})

    per_trip, saved_hmm, saved_clf, saved_routes = [], {}, {}, {}
    for t, trip in testing.items():
        prob = test_prob[t]
        points, starts, vectors = geometry(trip)
        graph = RoadGraph(trip["arcs"], starts, vectors)
        start = perf_counter()
        hmm_arcs, breaks = viterbi(points, trip["track"][:, 2], trip["candidates"], prob, graph, params)
        seconds = perf_counter() - start
        clf_arcs = classifier_only(trip, prob)
        hmm_route, unfilled = stitch(graph, hmm_arcs)
        clf_route, _ = stitch(graph, clf_arcs)
        c, h = score(trip, clf_arcs, prob), score(trip, hmm_arcs, prob)
        _, _, clf_f1 = route_overlap(clf_route, trip["route"], graph.length)
        hp, hr, hf = route_overlap(hmm_route, trip["route"], graph.length)
        per_trip.append({
            "trip_id": t, "observations": h["observations"],
            "clf_top1_accuracy": c["top1_accuracy"], "hmm_top1_accuracy": h["top1_accuracy"],
            "clf_macro_f1": c["macro_f1"], "hmm_macro_f1": h["macro_f1"],
            "clf_direct_continuity": c["direct_route_continuity"], "hmm_direct_continuity": h["direct_route_continuity"],
            "clf_arc_runs": c["distinct_arc_runs_predicted"], "hmm_arc_runs": h["distinct_arc_runs_predicted"],
            "reference_arc_runs": h["distinct_arc_runs_reference"],
            "clf_route_f1": clf_f1, "hmm_route_f1": hf, "hmm_route_precision": hp, "hmm_route_recall": hr,
            "breaks": breaks, "unfilled_gaps": unfilled, "viterbi_seconds": seconds,
        })
        saved_hmm[t], saved_clf[t] = hmm_arcs, clf_arcs
        saved_routes[t] = np.asarray(hmm_route)
        print(per_trip[-1], flush=True)
    save_csv(out / "hmm_vs_classifier_student_test.csv", per_trip)
    np.savez_compressed(out / "student_test_hmm_predictions.npz", **saved_hmm)
    np.savez_compressed(out / "student_test_classifier_predictions.npz", **saved_clf)
    np.savez_compressed(out / "student_test_hmm_routes.npz", **saved_routes)

    n = sum(r["observations"] for r in per_trip)
    summary = {k: sum(r[f"{k}"] * r["observations"] for r in per_trip) / n
               for k in ("clf_top1_accuracy", "hmm_top1_accuracy")}
    for k in ("clf_macro_f1", "hmm_macro_f1", "clf_direct_continuity", "hmm_direct_continuity",
              "clf_route_f1", "hmm_route_f1"):
        summary[k] = float(np.mean([r[k] for r in per_trip]))
    summary["clf_arc_runs"] = sum(r["clf_arc_runs"] for r in per_trip)
    summary["hmm_arc_runs"] = sum(r["hmm_arc_runs"] for r in per_trip)
    summary["reference_arc_runs"] = sum(r["reference_arc_runs"] for r in per_trip)
    summary["trips_hmm_top1_not_better"] = [r["trip_id"] for r in per_trip
                                            if r["hmm_top1_accuracy"] <= r["clf_top1_accuracy"]]
    write_json(out / "hmm_summary_student_test.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tuning-results", type=Path, default=HERE.parent / "pathsnap-tuning" / "results")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    run(**vars(parser.parse_args()))

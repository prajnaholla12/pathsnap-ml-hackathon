"""Server-side scoring against the hidden reference route.

Only main.predict calls this, after the prediction is finished.
Uses the same label alignment and metrics as pathsnap-training.
"""
from pathlib import Path

import numpy as np

from data import reference_labels
from train import score
from hmm import route_overlap


def score_trip(eval_dir, trip_id, track, arcs, result):
    path = Path(eval_dir) / trip_id / f"{trip_id}.route"
    out = {"viterbi_breaks": int(result["breaks"]), "unfilled_gaps": int(result["unfilled"]),
           "has_reference": path.exists(), "predicted_route_arcs": len(result["route"])}
    if not path.exists():
        return out | {k: None for k in ("hmm_top1_accuracy", "classifier_top1_accuracy", "hmm_macro_f1",
                                       "classifier_macro_f1", "hmm_direct_continuity",
                                       "classifier_direct_continuity", "candidate_miss_rate",
                                       "route_precision", "route_recall", "route_f1", "reference_route_arcs")}
    route = np.loadtxt(path, dtype=int, ndmin=1).ravel()
    if not len(route) or np.any((route < 0) | (route >= len(arcs))):
        raise ValueError("Reference route is empty or contains an out-of-range arc ID")
    labels, _ = reference_labels(result["points"], result["starts"], result["vectors"], route)
    trip = {"labels": labels, "candidates": result["candidates"], "arcs": arcs}
    hmm = score(trip, result["hmm_arcs"], result["prob"])
    clf = score(trip, result["clf_arcs"], result["prob"])
    precision, recall, f1 = route_overlap(result["route"], route, result["graph"].length)
    return out | {
        "hmm_top1_accuracy": hmm["top1_accuracy"], "classifier_top1_accuracy": clf["top1_accuracy"],
        "hmm_macro_f1": hmm["macro_f1"], "classifier_macro_f1": clf["macro_f1"],
        "hmm_direct_continuity": hmm["direct_route_continuity"],
        "classifier_direct_continuity": clf["direct_route_continuity"],
        "candidate_miss_rate": hmm["candidate_miss_rate"],
        "route_precision": precision, "route_recall": recall, "route_f1": f1,
        "predicted_route_arcs": len(result["route"]), "reference_route_arcs": int(len(route)),
    }

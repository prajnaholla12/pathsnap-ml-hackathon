"""Folium maps: EDA maps and one prediction map per student_test trip."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pathsnap-training"))
from train import read_trips  # noqa: E402
from maps import locations_map, trip_map  # noqa: E402


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    training, testing = read_trips(cache, "train"), read_trips(cache, "student_test")
    inventory = pd.read_csv(ROOT / "pathsnap-eda" / "results" / "eda_inventory.csv", dtype={"trip_id": str})
    locations_map(out / "eda_trip_locations.html", inventory)
    for partition, trips, t in (("train", training, "00000048"), ("train", training, "00000033"),
                                ("student_test", testing, "00000040")):
        trip = trips[t]
        trip_map(out / f"eda_{t}.html", f"{partition} {t}", trip["track"], trip["nodes"], trip["arcs"],
                 reference=trip["route"])

    hmm = ROOT / "pathsnap-hmm" / "results"
    routes = np.load(hmm / "student_test_hmm_routes.npz")
    clf = np.load(hmm / "student_test_classifier_predictions.npz")
    table = pd.read_csv(hmm / "hmm_vs_classifier_student_test.csv", dtype={"trip_id": str}).set_index("trip_id")
    for t, trip in testing.items():
        title = f"student_test {t}: HMM top-1 {100 * table.loc[t, 'hmm_top1_accuracy']:.1f}%"
        trip_map(out / f"prediction_{t}.html", title, trip["track"], trip["nodes"], trip["arcs"],
                 reference=trip["route"], classifier=clf[t], predicted=routes[t])
        print("map", t, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "maps")
    run(**vars(parser.parse_args()))

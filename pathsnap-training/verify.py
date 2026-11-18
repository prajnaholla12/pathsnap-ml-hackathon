"""Replay all saved estimators and check candidate search against direct distances."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from data import K, project, segment_distance
from train import models, predict, read_trips


def verify(cache, results):
    trips = read_trips(cache, "student_test")
    training = read_trips(cache, "train")
    folds = json.loads((results / "folds.json").read_text())
    all_validation = []
    assert not (training.keys() & trips.keys())
    for fold in folds:
        fit, validation = set(fold["fit_trip_ids"]), set(fold["validation_trip_ids"])
        assert not (fit & validation) and fit | validation == set(training)
        all_validation.extend(validation)
    assert sorted(all_validation) == sorted(training)
    with (results / "model_comparison.csv").open(newline="") as file:
        metrics = {row["model"]: row for row in csv.DictReader(file)}
    assert set(metrics) == set(models())
    for trip in trips.values():
        track, arcs = trip["track"], trip["arcs"]
        origin = track[:, :2].mean(axis=0)
        nodes, points = project(trip["nodes"], origin), project(track[:, :2], origin)
        starts, vectors = nodes[arcs[:, 0]], nodes[arcs[:, 1]] - nodes[arcs[:, 0]]
        for index in np.linspace(0, len(points) - 1, 12, dtype=int):
            distances, _ = segment_distance(points[index], starts, vectors)
            expected = np.lexsort((np.arange(len(arcs)), distances))[:K]
            np.testing.assert_array_equal(expected, trip["candidates"][index])
    for name in metrics:
        folder = results / name
        model = joblib.load(folder / "model.joblib")
        correct, count = 0, 0
        with np.load(folder / "student_test_predictions.npz") as saved:
            for trip_id, trip in trips.items():
                predicted, probabilities = predict(model, trip)
                np.testing.assert_array_equal(predicted, saved[trip_id])
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=5e-7)
                correct += np.count_nonzero(predicted == trip["labels"])
                count += len(predicted)
        np.testing.assert_allclose(correct / count, float(metrics[name]["student_top1_accuracy"]))
        print(f"{name}: verified {correct}/{count} correct predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(args.cache, args.results)

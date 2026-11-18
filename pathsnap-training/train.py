"""Compare five candidate classifiers with trip-grouped cross-validation."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from data import FEATURES, K


SEED = 42
FOLDS = 5


def models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED)),
        "gaussian_naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=30, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=15,
                                                max_samples=0.5, n_jobs=2, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31,
                                                                early_stopping=False, random_state=SEED),
    }


def read_trips(root, partition):
    trips = {}
    for path in sorted((root / partition).glob("*.npz")):
        with np.load(path) as archive:
            trips[path.stem] = dict(archive)
    expected = 80 if partition == "train" else 10
    if len(trips) != expected:
        raise ValueError(f"Expected {expected} {partition} trips, found {len(trips)}")
    return trips


def training_rows(trips):
    # When the reference arc is missing, retain the observation for evaluation only.
    masks = [trip["candidates"] == trip["labels"][:, None] for trip in trips]
    covered = [mask.any(axis=1) for mask in masks]
    x = np.concatenate([trip["x"][keep].reshape(-1, len(FEATURES))
                        for trip, keep in zip(trips, covered)])
    y = np.concatenate([mask[keep].ravel() for mask, keep in zip(masks, covered)])
    return x, y


def route_runs(arcs):
    return arcs[np.r_[True, arcs[1:] != arcs[:-1]]]


def predict(model, trip):
    x = trip["x"]
    raw = model.predict_proba(x.reshape(-1, len(FEATURES)))[:, 1].reshape(x.shape[:2])
    normalized = np.maximum(raw, 1e-12)
    normalized /= normalized.sum(axis=1, keepdims=True)
    chosen = np.argmax(raw, axis=1)
    rows = np.arange(len(x))
    return trip["candidates"][rows, chosen], normalized


def score(trip, prediction, probability):
    labels, candidates, arcs = trip["labels"], trip["candidates"], trip["arcs"]
    target = candidates == labels[:, None]
    covered = target.any(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="macro", zero_division=0)
    true_probability = (probability * target).sum(axis=1)
    connected = (prediction[:-1] == prediction[1:]) | (arcs[prediction[:-1], 1] == arcs[prediction[1:], 0])
    predicted_runs, reference_runs = route_runs(prediction), route_runs(labels)
    return {
        "observations": len(labels),
        "top1_accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "candidate_miss_rate": float(np.mean(~covered)),
        "candidate_covered_log_loss": float(-np.log(true_probability[covered]).mean()) if covered.any() else None,
        "direct_route_continuity": float(connected.mean()) if len(connected) else None,
        "distinct_arc_runs_predicted": len(predicted_runs),
        "distinct_arc_runs_reference": len(reference_runs),
    }


def evaluate(model, trips):
    rows, predictions = [], {}
    for trip_id, trip in trips.items():
        start = perf_counter()
        prediction, probabilities = predict(model, trip)
        elapsed = perf_counter() - start
        rows.append({"trip_id": trip_id, **score(trip, prediction, probabilities),
                     "model_inference_ms": elapsed * 1000})
        predictions[trip_id] = prediction
    return rows, predictions


def mean(rows, key):
    return float(np.mean([row[key] for row in rows if row[key] is not None]))


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def report_confusion(destination, predictions, trips):
    rows = []
    for trip_id, prediction in predictions.items():
        labels = trips[trip_id]["labels"]
        pairs, count = np.unique(np.column_stack((labels, prediction)), axis=0, return_counts=True)
        rows.extend({"trip_id": trip_id, "reference_arc": int(ref), "predicted_arc": int(pred),
                     "count": int(n)} for (ref, pred), n in zip(pairs, count))
    save_csv(destination, rows)


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    training, testing = read_trips(cache, "train"), read_trips(cache, "student_test")
    ids = np.array(list(training))
    folds = list(GroupKFold(n_splits=FOLDS).split(ids, groups=ids))
    write_json(out / "folds.json", [
        {"fold": fold, "fit_trip_ids": ids[fit].tolist(), "validation_trip_ids": ids[val].tolist()}
        for fold, (fit, val) in enumerate(folds, 1)
    ])
    write_json(out / "run_config.json", {"seed": SEED, "candidate_count": K, "features": FEATURES,
               "sklearn_version": sklearn.__version__, "numpy_version": np.__version__})
    inventory = []
    for partition, trips in (("train", training), ("student_test", testing)):
        for trip_id, trip in trips.items():
            gaps = np.diff(trip["track"][:, 2])
            inventory.append({"partition": partition, "trip_id": trip_id, "observations": len(trip["labels"]),
                              "invalid_rows": int(trip["invalid_rows"]), "reordered": bool(trip["reordered"]),
                              "nodes": len(trip["nodes"]), "arcs": len(trip["arcs"]),
                              "reference_arcs": len(trip["route"]), "maximum_gap_s": float(np.max(gaps, initial=0)),
                              "median_reference_alignment_m": float(np.median(trip["alignment_distance"])),
                              "candidate_miss_rate": float(np.mean(~(trip["candidates"] == trip["labels"][:, None]).any(axis=1)))})
    save_csv(out / "dataset_inventory.csv", inventory)
    full_x, full_y = training_rows(list(training.values()))
    summary = []
    for name in models():
        folder = out / name
        folder.mkdir(exist_ok=True)
        cv_rows = []
        for fold, (fit, val) in enumerate(folds, 1):
            fit_x, fit_y = training_rows([training[i] for i in ids[fit]])
            model = models()[name]
            model.fit(fit_x, fit_y)
            rows, _ = evaluate(model, {i: training[i] for i in ids[val]})
            cv_rows.extend({"fold": fold, **row} for row in rows)
            print(name, "fold", fold, "macro F1", round(mean(rows, "macro_f1"), 4), flush=True)
        save_csv(folder / "cross_validation.csv", cv_rows)
        model = models()[name]
        start = perf_counter()
        model.fit(full_x, full_y)
        fit_seconds = perf_counter() - start
        train_rows, _ = evaluate(model, training)
        test_rows, predicted = evaluate(model, testing)
        save_csv(folder / "training.csv", train_rows)
        save_csv(folder / "student_test.csv", test_rows)
        report_confusion(folder / "student_test_confusion.csv", predicted, testing)
        np.savez_compressed(folder / "student_test_predictions.npz", **predicted)
        joblib.dump(model, folder / "model.joblib", compress=3)
        write_json(folder / "config.json", {"model": repr(model), "parameters": model.get_params(),
                                            "feature_names": FEATURES, "candidate_count": K, "seed": SEED})
        observations = sum(row["observations"] for row in test_rows)
        row = {"model": name, "cv_macro_f1": mean(cv_rows, "macro_f1"),
               "train_top1_accuracy": mean(train_rows, "top1_accuracy"),
               "train_macro_f1": mean(train_rows, "macro_f1"),
               "student_top1_accuracy": sum(row["observations"] * row["top1_accuracy"] for row in test_rows) / observations,
               "student_macro_f1": mean(test_rows, "macro_f1"),
               "student_macro_precision": mean(test_rows, "macro_precision"),
               "student_macro_recall": mean(test_rows, "macro_recall"),
               "student_candidate_miss_rate": sum(row["candidate_miss_rate"] * row["observations"] for row in test_rows) / observations,
               "student_covered_log_loss": mean(test_rows, "candidate_covered_log_loss"),
               "student_direct_continuity": mean(test_rows, "direct_route_continuity"),
               "mean_inference_ms_per_trip": mean(test_rows, "model_inference_ms"),
               "full_fit_seconds": fit_seconds}
        summary.append(row)
        save_csv(out / "model_comparison.csv", summary)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.cache, args.out)

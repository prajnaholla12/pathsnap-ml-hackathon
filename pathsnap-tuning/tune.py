"""Bayesian hyperparameter search for the Histogram Gradient Boosting baseline.

Uses Optuna's TPE sampler. The objective is the same cross-validation Macro F1
as pathsnap-training/train.py, on the same trip folds (results/folds.json).
student_test is only used after the search, to compare tuned and untuned models.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

TRAINING = Path(__file__).resolve().parent.parent / "pathsnap-training"
sys.path.insert(0, str(TRAINING))
from data import FEATURES  # noqa: E402
from train import SEED, evaluate, mean, models, predict, read_trips, save_csv, training_rows, write_json  # noqa: E402


def make(params):
    return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **params)


def cross_validate(params, training, folds, keep_probabilities=False):
    rows, probabilities = [], {}
    for fold in folds:
        fit_x, fit_y = training_rows([training[i] for i in fold["fit_trip_ids"]])
        model = make(params) if params is not None else models()["hist_gradient_boosting"]
        model.fit(fit_x, fit_y)
        validation = {i: training[i] for i in fold["validation_trip_ids"]}
        fold_rows, _ = evaluate(model, validation)
        rows.extend({"fold": fold["fold"], **row} for row in fold_rows)
        if keep_probabilities:
            for trip_id, trip in validation.items():
                probabilities[trip_id] = predict(model, trip)[1].astype(np.float32)
    return rows, probabilities


def summarise(name, params, cv_rows, model, fit_seconds, testing):
    test_rows, predicted = evaluate(model, testing)
    observations = sum(r["observations"] for r in test_rows)
    return {
        "model": name,
        "cv_macro_f1": mean(cv_rows, "macro_f1"),
        "cv_macro_f1_fold_std": float(np.std([np.mean([r["macro_f1"] for r in cv_rows if r["fold"] == f])
                                              for f in sorted({r["fold"] for r in cv_rows})])),
        "cv_top1_accuracy": sum(r["observations"] * r["top1_accuracy"] for r in cv_rows)
                            / sum(r["observations"] for r in cv_rows),
        "student_top1_accuracy": sum(r["observations"] * r["top1_accuracy"] for r in test_rows) / observations,
        "student_macro_f1": mean(test_rows, "macro_f1"),
        "student_covered_log_loss": mean(test_rows, "candidate_covered_log_loss"),
        "student_direct_continuity": mean(test_rows, "direct_route_continuity"),
        "mean_inference_ms_per_trip": mean(test_rows, "model_inference_ms"),
        "full_fit_seconds": fit_seconds,
        "parameters": json.dumps(params) if params else "train.py defaults",
    }, test_rows, predicted


def run(cache, training_results, out, trials):
    out.mkdir(parents=True, exist_ok=True)
    training, testing = read_trips(cache, "train"), read_trips(cache, "student_test")
    folds = json.loads((training_results / "folds.json").read_text())

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 400, step=25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 400, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }
        start = perf_counter()
        rows, _ = cross_validate(params, training, folds)
        score = mean(rows, "macro_f1")
        trial.set_user_attr("cv_top1_accuracy", sum(r["observations"] * r["top1_accuracy"] for r in rows)
                            / sum(r["observations"] for r in rows))
        trial.set_user_attr("seconds", perf_counter() - start)
        print(f"trial {trial.number}: cv macro F1 {score:.4f} ({perf_counter() - start:.0f}s) {params}", flush=True)
        return score

    study = optuna.create_study(direction="maximize", study_name="hgb_cv_macro_f1",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
                                storage=f"sqlite:///{(out / 'optuna_study.db').as_posix()}", load_if_exists=True)
    if not study.trials:
        # Start near the baseline; L2 is the search lower bound, not baseline 0.
        study.enqueue_trial({"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31,
                             "min_samples_leaf": 20, "l2_regularization": 1e-4, "max_features": 1.0})
    done = sum(t.state.is_finished() for t in study.trials)
    if trials > done:
        study.optimize(objective, n_trials=trials - done)

    trial_rows = [{"trial": t.number, "state": t.state.name, "cv_macro_f1": t.value,
                   "cv_top1_accuracy": t.user_attrs.get("cv_top1_accuracy"),
                   "seconds": t.user_attrs.get("seconds"), **t.params} for t in study.trials]
    save_csv(out / "trials.csv", trial_rows)
    best = study.best_params
    write_json(out / "best_params.json", {"best_params": best, "best_cv_macro_f1": study.best_value,
                                          "best_trial": study.best_trial.number,
                                          "completed_trials": sum(t.state.name == "COMPLETE" for t in study.trials)})

    full_x, full_y = training_rows(list(training.values()))
    comparison = []
    for name, params in (("hist_gradient_boosting_untuned", None), ("hist_gradient_boosting_tuned", best)):
        cv_rows, oof = cross_validate(params, training, folds, keep_probabilities=params is not None)
        model = make(params) if params is not None else models()["hist_gradient_boosting"]
        start = perf_counter()
        model.fit(full_x, full_y)
        fit_seconds = perf_counter() - start
        row, test_rows, predicted = summarise(name, params, cv_rows, model, fit_seconds, testing)
        comparison.append(row)
        print(json.dumps(row), flush=True)
        if params is not None:
            folder = out / "tuned_model"
            folder.mkdir(exist_ok=True)
            joblib.dump(model, folder / "model.joblib", compress=3)
            save_csv(folder / "cross_validation.csv", cv_rows)
            save_csv(folder / "student_test.csv", test_rows)
            np.savez_compressed(folder / "student_test_predictions.npz", **predicted)
            np.savez_compressed(folder / "train_oof_probabilities.npz", **oof)
            np.savez_compressed(folder / "student_test_probabilities.npz",
                                **{t: predict(model, trip)[1].astype(np.float32) for t, trip in testing.items()})
            write_json(folder / "config.json", {"model": repr(model), "parameters": model.get_params(),
                                                "feature_names": FEATURES, "seed": SEED})
    save_csv(out / "tuned_vs_untuned.csv", comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--training-results", type=Path, default=TRAINING / "results")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with threadpool_limits(limits=2):
        run(args.cache, args.training_results, args.out, args.trials)

"""All comparison figures for the report, from the CSV/NPZ outputs of the other stages."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pathsnap-training"))
from train import read_trips, training_rows  # noqa: E402

TRAIN = ROOT / "pathsnap-training" / "results"
TUNE = ROOT / "pathsnap-tuning" / "results"
HMM = ROOT / "pathsnap-hmm" / "results"
ORDER = ["logistic_regression", "gaussian_naive_bayes", "decision_tree", "random_forest", "hist_gradient_boosting"]
SHORT = {"logistic_regression": "LogReg", "gaussian_naive_bayes": "GaussianNB", "decision_tree": "DecisionTree",
         "random_forest": "RandomForest", "hist_gradient_boosting": "HistGB"}
COLOURS = ["#6b8fb8", "#8c8c8c", "#b5835a", "#5a9e6f", "#c0504d"]


def fold_scores():
    rows = []
    for m in ORDER:
        cv = pd.read_csv(TRAIN / m / "cross_validation.csv")
        for fold, g in cv.groupby("fold"):
            rows.append({"model": m, "fold": fold, "macro_f1": g.macro_f1.mean(),
                         "top1": np.average(g.top1_accuracy, weights=g.observations)})
    return pd.DataFrame(rows)


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    mc = pd.read_csv(TRAIN / "model_comparison.csv").set_index("model").loc[ORDER]
    folds = fold_scores()
    folds.to_csv(out / "cv_fold_scores.csv", index=False)
    x = np.arange(len(ORDER))
    names = [SHORT[m] for m in ORDER]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for k, (col, title) in enumerate((("macro_f1", "CV Macro F1"), ("top1", "CV top-1 accuracy"))):
        g = folds.groupby("model")[col].agg(["mean", "std"]).loc[ORDER]
        ax[k].bar(x, g["mean"], yerr=g["std"], color=COLOURS, capsize=4)
        for i, m in enumerate(ORDER):
            f = folds[folds.model == m][col]
            ax[k].scatter(i + np.linspace(-0.15, 0.15, len(f)), f, s=12, c="k", zorder=3)
        ax[k].set_xticks(x, names); ax[k].set_ylim(0.55, 0.8 if k == 0 else 0.95)
        ax[k].set_title(f"{title}, 5-fold trip-grouped CV (dots = folds)")
    plt.tight_layout(); plt.savefig(out / "models_cv.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for k, (col, title) in enumerate((("train_macro_f1", "train (refit)"), ("cv_macro_f1", "CV"),
                                      ("student_macro_f1", "student_test"))):
        ax[0].bar(x + (k - 1) * 0.27, mc[col], 0.27, label=title, color=["#9db7d5", "#3b6ea8", "#d9822b"][k])
    ax[0].set_xticks(x, names, rotation=15); ax[0].set_ylim(0.6, 0.8); ax[0].legend(fontsize=8)
    ax[0].set_title("Macro F1: train vs CV vs student_test")
    ax[1].bar(x, mc.student_top1_accuracy * 100, color=COLOURS)
    for i, v in enumerate(mc.student_top1_accuracy):
        ax[1].text(i, v * 100 + 0.05, f"{v * 100:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(x, names, rotation=15); ax[1].set_ylim(84, 86)
    ax[1].set_title("student_test top-1 accuracy (%)")
    ax[2].bar(x, mc.student_covered_log_loss, color=COLOURS)
    ax[2].set_xticks(x, names, rotation=15); ax[2].set_title("student_test covered log loss")
    plt.tight_layout(); plt.savefig(out / "models_student_test.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(x, mc.full_fit_seconds, color=COLOURS); ax[0].set_xticks(x, names, rotation=15)
    ax[0].set_ylabel("seconds"); ax[0].set_title("Training time (fit on all 80 trips)")
    ax[1].bar(x, mc.mean_inference_ms_per_trip, color=COLOURS); ax[1].set_xticks(x, names, rotation=15)
    ax[1].set_ylabel("ms"); ax[1].set_title("Inference time per student_test trip")
    plt.tight_layout(); plt.savefig(out / "models_time.png", dpi=130); plt.close()

    # boosting curves (log loss per iteration) for untuned and tuned HistGB
    training, testing = read_trips(cache, "train"), read_trips(cache, "student_test")
    ids = sorted(training)[::5]
    tx, ty = training_rows([training[i] for i in ids])
    vx, vy = training_rows(list(testing.values()))
    curves = {}
    for label, path in (("untuned", TRAIN / "hist_gradient_boosting" / "model.joblib"),
                        ("tuned", TUNE / "tuned_model" / "model.joblib")):
        model = joblib.load(path)
        tr = [log_loss(ty, p[:, 1], labels=[0, 1]) for p in model.staged_predict_proba(tx)]
        te = [log_loss(vy, p[:, 1], labels=[0, 1]) for p in model.staged_predict_proba(vx)]
        curves[label] = (tr, te)
        pd.DataFrame({"iteration": np.arange(1, len(tr) + 1), "train_log_loss": tr, "student_test_log_loss": te}) \
            .to_csv(out / f"boosting_curve_{label}.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, label in zip(ax, ("untuned", "tuned")):
        tr, te = curves[label]
        a.plot(np.arange(1, len(tr) + 1), tr, c="#3b6ea8", label="train (16 of the 80 trips)")
        a.plot(np.arange(1, len(te) + 1), te, c="#c0504d", label="student_test (logged only)")
        a.set_xlabel("boosting iteration"); a.set_ylabel("candidate log loss")
        a.set_title(f"HistGB {label}: loss per iteration"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "boosting_curves.png", dpi=130); plt.close()

    trials = pd.read_csv(TUNE / "trials.csv")
    trials = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].scatter(trials.trial, trials.cv_macro_f1, s=18, c="#3b6ea8", label="trial")
    ax[0].plot(trials.trial, trials.cv_macro_f1.cummax(), c="#c0504d", label="best so far")
    ax[0].axhline(trials.cv_macro_f1.iloc[0], ls="--", c="gray", lw=1)
    ax[0].set_xlabel("trial (0 = untuned settings)"); ax[0].set_ylabel("CV Macro F1"); ax[0].legend()
    ax[0].set_title("Optuna TPE search history")
    sc = ax[1].scatter(trials.learning_rate, trials.cv_macro_f1, c=trials.max_iter, cmap="viridis", s=24)
    ax[1].set_xscale("log"); ax[1].set_xlabel("learning rate (colour = max_iter)"); ax[1].set_ylabel("CV Macro F1")
    plt.colorbar(sc, ax=ax[1]); ax[1].set_title("Learning rate vs score")
    plt.tight_layout(); plt.savefig(out / "optuna_history.png", dpi=130); plt.close()

    tu = pd.read_csv(TUNE / "tuned_vs_untuned.csv").set_index("model")
    metrics = [("cv_macro_f1", "CV Macro F1"), ("student_macro_f1", "test Macro F1"),
               ("student_top1_accuracy", "test top-1")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    for k, (name, row) in enumerate(tu.iterrows()):
        vals = [row[m] for m, _ in metrics]
        ax[0].bar(np.arange(3) + (k - 0.5) * 0.4, vals, 0.4, label=name.replace("hist_gradient_boosting_", ""),
                  color=["#9db7d5", "#c0504d"][k])
        for j, v in enumerate(vals):
            ax[0].text(j + (k - 0.5) * 0.4, v + 0.003, f"{v:.4f}", ha="center", fontsize=7)
    ax[0].set_xticks(range(3), [n for _, n in metrics]); ax[0].set_ylim(0.65, 0.9); ax[0].legend()
    ax[0].set_title("HistGB untuned vs tuned")
    for a, col, title in ((ax[1], "student_covered_log_loss", "test log loss"),
                          (ax[2], "mean_inference_ms_per_trip", "ms per trip")):
        a.bar(["untuned", "tuned"], tu[col], color=["#9db7d5", "#c0504d"])
        for i, v in enumerate(tu[col]):
            a.text(i, v, f"{v:.3f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        a.set_title(title)
    plt.tight_layout(); plt.savefig(out / "tuned_vs_untuned.png", dpi=130); plt.close()

    grid = pd.read_csv(HMM / "decoder_grid_train.csv")
    best = grid.loc[grid.macro_f1.idxmax()]
    piv = grid[grid.beta_gap == best.beta_gap].pivot(index="w_emit", columns="beta", values="macro_f1")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("beta (m)"); ax.set_ylabel("emission weight")
    ax.set_title(f"Decoder grid on 24 train trips, Macro F1 (beta_gap={best.beta_gap:g})", fontsize=9)
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(out / "hmm_grid.png", dpi=130); plt.close()

    per = pd.read_csv(HMM / "hmm_vs_classifier_student_test.csv", dtype={"trip_id": str})
    xx = np.arange(len(per))
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.9))
    for a, (c, h, title) in zip(ax, (("clf_top1_accuracy", "hmm_top1_accuracy", "Top-1 accuracy"),
                                     ("clf_macro_f1", "hmm_macro_f1", "Macro F1"),
                                     ("clf_direct_continuity", "hmm_direct_continuity", "Direct route continuity"))):
        a.bar(xx - 0.2, per[c], 0.4, label="classifier only", color="#e67e22")
        a.bar(xx + 0.2, per[h], 0.4, label="classifier + HMM", color="#d62728")
        a.set_xticks(xx, per.trip_id.str[-3:]); a.set_xlabel("student_test trip (last 3 digits)")
        a.set_title(title); a.set_ylim(0.5, 1.0)
    ax[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "hmm_vs_classifier.png", dpi=130); plt.close()

    # where errors happen: speed bins and reverse-direction picks
    clf = np.load(HMM / "student_test_classifier_predictions.npz")
    hmm = np.load(HMM / "student_test_hmm_predictions.npz")
    rows = []
    for t, trip in testing.items():
        speed = np.expm1(trip["x"][:, 0, 2])
        arcs, labels = trip["arcs"], trip["labels"]
        for name, pred in (("classifier only", clf[t]), ("classifier + HMM", hmm[t])):
            wrong = pred != labels
            reverse = (arcs[pred, 0] == arcs[labels, 1]) & (arcs[pred, 1] == arcs[labels, 0])
            rows.append(pd.DataFrame({"decoder": name, "trip_id": t, "speed": speed, "correct": ~wrong,
                                      "reverse_error": wrong & reverse}))
    err = pd.concat(rows)
    err["speed_bin"] = pd.cut(err.speed, [-1, 1, 5, 15, 1e9], labels=["< 1 m/s", "1-5", "5-15", "> 15"])
    by_speed = err.groupby(["decoder", "speed_bin"], observed=True).correct.mean().unstack(0)
    counts = err[err.decoder == "classifier only"].speed_bin.value_counts().sort_index()
    rev = err.groupby("decoder").apply(lambda g: g.reverse_error.sum() / max(1, (~g.correct).sum()),
                                       include_groups=False)
    summary = {"accuracy_by_speed": by_speed.round(4).to_dict(), "readings_per_speed_bin": counts.to_dict(),
               "reverse_direction_share_of_errors": rev.round(4).to_dict()}
    (out / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=str))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    by_speed.plot.bar(ax=ax, color=["#d62728", "#e67e22"], rot=0)
    ax.set_ylabel("top-1 accuracy"); ax.set_xlabel("GPS speed (readings per bin: "
                                                   + ", ".join(str(v) for v in counts.values) + ")")
    ax.set_ylim(0.3, 1.0); ax.set_title("student_test accuracy by speed (tuned HistGB)")
    plt.tight_layout(); plt.savefig(out / "error_by_speed.png", dpi=130); plt.close()
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    run(**vars(parser.parse_args()))

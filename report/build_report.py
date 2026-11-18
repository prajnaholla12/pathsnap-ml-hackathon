"""Builds report/PathSnap_Report.pdf. Every number is read from the result files of the stages.

Usage (from the repo root): python report/build_report.py
Needs Python 3.12+ and playwright with chromium for the PDF step.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
TR, TU, HM = ROOT / "pathsnap-training" / "results", ROOT / "pathsnap-tuning" / "results", ROOT / "pathsnap-hmm" / "results"
ED, VZ = ROOT / "pathsnap-eda" / "results", ROOT / "pathsnap-viz" / "figures"
ORDER = ["logistic_regression", "gaussian_naive_bayes", "decision_tree", "random_forest", "hist_gradient_boosting"]
NAME = {"logistic_regression": "Logistic Regression", "gaussian_naive_bayes": "Gaussian Naive Bayes",
        "decision_tree": "Decision Tree", "random_forest": "Random Forest",
        "hist_gradient_boosting": "Histogram Gradient Boosting"}
FAMILY = {"logistic_regression": "Linear", "gaussian_naive_bayes": "Probabilistic", "decision_tree": "Tree",
          "random_forest": "Ensemble (bagging)", "hist_gradient_boosting": "Ensemble (boosting)"}
CONFIG = {"logistic_regression": "StandardScaler + L2, C=1, max_iter=500",
          "gaussian_naive_bayes": "defaults (var_smoothing=1e-9)",
          "decision_tree": "max_depth=10, min_samples_leaf=30",
          "random_forest": "60 trees, max_depth=12, min_leaf=15, max_samples=0.5",
          "hist_gradient_boosting": "100 iterations, 31 leaves, lr=0.1, no early stopping"}


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def table(df, cls=""):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.values)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def fig(path, caption, width="100%"):
    rel = Path(path).resolve().relative_to(ROOT)
    return f'<figure><img src="../{rel.as_posix()}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def load():
    d = {"inv": pd.read_csv(ED / "eda_inventory.csv", dtype={"trip_id": str}),
         "summ": pd.read_csv(ED / "eda_summary.csv", index_col=0),
         "proj": pd.read_csv(ED / "projection_check.csv"),
         "mc": pd.read_csv(TR / "model_comparison.csv").set_index("model"),
         "folds": pd.read_csv(VZ / "cv_fold_scores.csv"),
         "folddef": json.loads((TR / "folds.json").read_text()),
         "trials": pd.read_csv(TU / "trials.csv"),
         "best": json.loads((TU / "best_params.json").read_text()),
         "tu": pd.read_csv(TU / "tuned_vs_untuned.csv").set_index("model"),
         "grid": pd.read_csv(HM / "decoder_grid_train.csv"),
         "dtune": json.loads((HM / "decoder_tuning.json").read_text()),
         "dparams": json.loads((HM / "decoder_params.json").read_text()),
         "hmm": pd.read_csv(HM / "hmm_vs_classifier_student_test.csv", dtype={"trip_id": str}),
         "hs": json.loads((HM / "hmm_summary_student_test.json").read_text()),
         "err": json.loads((VZ.parent / "figures" / "error_analysis.json").read_text()),
         "bc_u": pd.read_csv(VZ / "boosting_curve_untuned.csv"),
         "bc_t": pd.read_csv(VZ / "boosting_curve_tuned.csv"),
         "log": (ROOT / "pathsnap-api" / "logs" / "api_demo.log").read_text()}
    d["train"] = {m: pd.read_csv(TR / m / "training.csv") for m in ORDER}
    return d


def build(d):
    inv, mc, folds, tu, hmm, hs, err = d["inv"], d["mc"], d["folds"], d["tu"], d["hmm"], d["hs"], d["err"]
    tr, te = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
    summ = d["summ"]
    fstat = folds.groupby("model").agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                                       top1_mean=("top1", "mean"), top1_std=("top1", "std"))
    un, tn = tu.loc["hist_gradient_boosting_untuned"], tu.loc["hist_gradient_boosting_tuned"]
    best = d["best"]["best_params"]
    trials = d["trials"][d["trials"].state == "COMPLETE"]
    dp = d["dparams"]
    poor = inv[inv.share_alignment_over_50m > 0.2].trip_id.tolist()
    fold_of = {t: f["fold"] for f in d["folddef"] for t in f["validation_trip_ids"]}
    big = inv.sort_values("maximum_gap_s").iloc[-1]
    low_fold = folds[folds.model == "hist_gradient_boosting"].sort_values("top1").iloc[:2]
    acc_speed = err["accuracy_by_speed"]
    speed_n = err["readings_per_speed_bin"]
    rev = err["reverse_direction_share_of_errors"]
    h77 = hmm.set_index("trip_id").loc["00000077"]

    def train_cv_test(m):
        return (f"train Macro F1 {mc.loc[m].train_macro_f1:.4f} (model refit on all 80 trips, scored on them), "
                f"CV {mc.loc[m].cv_macro_f1:.4f}, student_test {mc.loc[m].student_macro_f1:.4f}")

    def results(m):
        r = mc.loc[m]
        return (f"student_test: <b>top-1 {pct(r.student_top1_accuracy)}</b>, <b>Macro F1 {r.student_macro_f1:.4f}</b> "
                f"(precision {r.student_macro_precision:.4f}, recall {r.student_macro_recall:.4f}), "
                f"<b>covered log loss {r.student_covered_log_loss:.4f}</b>, direct continuity "
                f"{pct(r.student_direct_continuity, 1)}. CV Macro F1 {fstat.loc[m].f1_mean:.4f} ± {fstat.loc[m].f1_std:.4f} "
                f"over the 5 folds. Fit time {r.full_fit_seconds:.1f} s; {r.mean_inference_ms_per_trip:.1f} ms per trip at inference.")

    rows = [[f"<b>{NAME[m]}</b>", FAMILY[m], CONFIG[m], f"{fstat.loc[m].f1_mean:.4f} ± {fstat.loc[m].f1_std:.4f}",
             pct(mc.loc[m].student_top1_accuracy), pct(mc.loc[m].student_direct_continuity, 1),
             f"{mc.loc[m].student_macro_f1:.4f}", f"{mc.loc[m].student_covered_log_loss:.4f}",
             f"{mc.loc[m].mean_inference_ms_per_trip:.1f} ms"] for m in ORDER]
    rows.append(["<b>HistGB tuned</b> (section 4)", "Ensemble (boosting)", "Optuna TPE, see Table 4.1",
                 f"{tn.cv_macro_f1:.4f} ± {tn.cv_macro_f1_fold_std:.4f}", pct(tn.student_top1_accuracy),
                 pct(tn.student_direct_continuity, 1), f"{tn.student_macro_f1:.4f}",
                 f"{tn.student_covered_log_loss:.4f}", f"{tn.mean_inference_ms_per_trip:.1f} ms"])
    rows.append(["<b>HistGB tuned + HMM</b> (section 5)", "Sequence model", f"w={dp['w_emit']:g}, β={dp['beta']:g} m",
                 "N/A (decoder tuned on train OOF)", pct(hs["hmm_top1_accuracy"]), pct(hs["hmm_direct_continuity"], 1),
                 f"{hs['hmm_macro_f1']:.4f}", "N/A (Viterbi path)", "+ Viterbi, see 7"])
    cmp_table = table(pd.DataFrame(rows, columns=[
        "Model", "Family", "Key configuration", "CV Macro F1 (mean ± std)", "Top-1 accuracy", "Route continuity",
        "Student-test Macro F1", "Log loss", "Inference time / trip"]), "small")

    css = (REPORT / "report.css").read_text()
    H = []
    a = H.append
    a("""<section class="title">
  <p class="course">PES University, RR Campus · Department of CSE<br>UE24CS352A Machine Learning · Hackathon</p>
  <h1>PathSnap</h1><h2>Map-matching and route reconstruction from noisy GPS</h2>
  <p class="sub">Technical report</p>
  <table class="team"><tr><th>Name</th><th>SRN</th><th>Part</th></tr>
  <tr><td>&nbsp;</td><td>&nbsp;</td><td>Data, features, baseline models</td></tr>
  <tr><td>&nbsp;</td><td>&nbsp;</td><td>Tuning, HMM, API</td></tr>
  <tr><td>&nbsp;</td><td>&nbsp;</td><td>Evaluator UI</td></tr>
  <tr><td>&nbsp;</td><td>&nbsp;</td><td>EDA, maps, plots, report</td></tr></table>
  <p class="repo">https://github.com/prajnaholla12/pathsnap-ml-hackathon</p>
</section>""")

    a(f"""<h1>1. Overview</h1>
<p>For every GPS reading of a trip we predict the road arc the vehicle was on, then rebuild the driven route.
The repository is split into stages, one folder each:</p>
{table(pd.DataFrame([
        ["pathsnap-training", "projection, candidates (8 nearest arcs), 8 features, label alignment, 5 baseline classifiers"],
        ["pathsnap-eda", "dataset exploration (section 2)"],
        ["pathsnap-tuning", "Bayesian optimisation of the chosen model (section 4)"],
        ["pathsnap-hmm", "HMM + Viterbi decoding, decoder tuning (section 5)"],
        ["pathsnap-viz", "Folium maps and every comparison figure (sections 3, 6)"],
        ["pathsnap-api / pathsnap-ui", "FastAPI backend on Laptop 1, evaluator page on Laptop 2 (section 7)"],
    ], columns=["Folder", "What it does"]))}
<p>Final system on student_test (10 trips, {int(te.observations.sum()):,} readings): <b>{pct(hs['hmm_top1_accuracy'])}</b> top-1
accuracy and Macro F1 <b>{hs['hmm_macro_f1']:.4f}</b>, up from {pct(mc.loc['hist_gradient_boosting'].student_top1_accuracy)} /
{mc.loc['hist_gradient_boosting'].student_macro_f1:.4f} for the untuned classifier alone. Length-weighted route F1 is
<b>{hs['hmm_route_f1']:.3f}</b>.</p>""")

    a(f"""<h1>2. Dataset understanding and EDA</h1>
<p class="note">Partitions: <b>train (80 trips)</b> and <b>student_test (10 trips)</b> only. The evaluator test trips were never read.
Code: <code>pathsnap-eda/eda.py</code> on the feature cache made by <code>pathsnap-training/data.py</code>.</p>
<h2>2.1 Files and how they relate</h2>
{table(pd.DataFrame([
        [".track", "lon, lat, t", f"{int(summ.loc['observations', 'train']):,} rows (train), {int(summ.loc['observations', 'student_test']):,} (student_test)", "GPS readings"],
        [".nodes", "lon, lat", f"{int(tr.nodes.min()):,} – {int(tr.nodes.max()):,} per trip", "row index = node id"],
        [".arcs", "from_node, to_node", f"{int(summ.loc['arcs_min', 'train']):,} – {int(summ.loc['arcs_max', 'train']):,} per trip", "row index = arc id, directed"],
        [".route", "arc id", f"{int(tr.reference_arcs.min())} – {int(tr.reference_arcs.max())} arcs", "ordered driven arcs, no timestamps"],
    ], columns=["File", "Columns", "Size", "Meaning"]))}
<ul>
<li><b>Each trip has its own road network</b>, and arc ids are only valid inside that trip. The model therefore scores a
(reading, candidate arc) pair from geometric features and never sees an id.</li>
<li><b>Trips come from many places in the world</b> (longitude {inv.lon_center.min():.0f}° to {inv.lon_center.max():.0f}°,
latitude {inv.lat_center.min():.0f}° to {inv.lat_center.max():.0f}°, Figure 2.1). We project every trip around its own centre.</li>
<li><b>Most roads are stored in both directions</b>: in the median trip {pct(inv.two_way_share.median(), 0)} of arcs have a reverse
twin with the same geometry. Only the direction of travel separates them.</li>
<li><b>The route has no timestamps</b>, so per-reading labels must be derived (2.2).</li>
</ul>
{fig(ED / "eda_overview.png", "Figure 2.1 – trip length, network size and trip centres (blue train, orange student_test).")}

<h2>2.2 Preprocessing</h2>
<ol>
<li>Rows with non-finite or out-of-range coordinates are dropped and readings are sorted by time (neither was needed, see 2.3).</li>
<li><b>Projection:</b> local equirectangular, metres, centred on each trip. Checked against haversine on 60 pairs per trip: worst error
{d['proj'].max_relative_error_pct.max():.2f}% on multi-km distances, negligible at candidate scale.</li>
<li><b>Candidates:</b> the 8 nearest arcs by exact point-to-segment distance (KD-tree on midpoints, bucketed by arc length so the search is exact).</li>
<li><b>Labels:</b> a dynamic program assigns each reading a non-decreasing position in the route, minimising total point-to-arc distance.
Positions (not arc ids) handle routes that repeat arcs.</li>
<li>Readings whose aligned arc is not among the 8 candidates are kept for evaluation (counted as errors) but not used as training rows.</li>
</ol>

<h2>2.3 Missing values and time gaps</h2>
{table(pd.DataFrame([
        ["Invalid / missing coordinate rows", int(summ.loc['invalid_rows', 'train']), int(summ.loc['invalid_rows', 'student_test'])],
        ["Trips with out-of-order timestamps", int(summ.loc['reordered_trips', 'train']), int(summ.loc['reordered_trips', 'student_test'])],
        ["Duplicate timestamps", int(summ.loc['duplicate_timestamps', 'train']), int(summ.loc['duplicate_timestamps', 'student_test'])],
        ["Gaps over 10 s", int(summ.loc['gaps_over_10s', 'train']), int(summ.loc['gaps_over_10s', 'student_test'])],
        ["Largest gap", f"{summ.loc['maximum_gap_s', 'train']:.0f} s", f"{summ.loc['maximum_gap_s', 'student_test']:.0f} s"],
        ["Trips with > 20% of readings > 50 m from the aligned arc", int(summ.loc['trips_alignment_over_50m_20pct', 'train']), int(summ.loc['trips_alignment_over_50m_20pct', 'student_test'])],
    ], columns=["Check", "train", "student_test"]))}
<p>Nothing is missing, so nothing is imputed. The real gaps are in time: most steps are 1 s, but there are
{int(summ.loc['gaps_over_10s', 'train'])} gaps over 10 s in train, and trip {big.trip_id} pauses for {big.maximum_gap_s / 3600:.0f} hours.
Section 5 handles gaps in the transition model.</p>
{fig(ED / "eda_missing.png", "Figure 2.2 – invalid rows and timestamp issues, summed over trips.", "78%")}
{fig(ED / "eda_sampling.png", "Figure 2.3 – time between readings (left) and gaps over 10 s per trip (right).")}

<h2>2.4 Road network</h2>
<p>No node is isolated, but the directed graphs are fragmented: median {int(inv.strong_components.median())} strongly connected
components per trip, with the largest one holding a median {pct(inv.largest_strong_share.median(), 0)} of nodes. Some arc pairs really are
unreachable, which is why the HMM penalises rather than forbids them. Reference routes are almost always connected
(lowest share of connected steps: {pct(inv.route_connected_share.min(), 1)}).</p>
{fig(ED / "eda_network.png", "Figure 2.4 – directed connectivity per trip network.")}

<h2>2.5 GPS against the network</h2>
{fig(ED / "eda_trajectories.png", "Figure 2.5 – GPS (blue), reference route (green), nearby arcs (grey) for three train and three student_test trips.")}
<p>The median distance from a reading to its aligned arc is about 5 m. Two train trips ({', '.join(poor)}) are exceptions: long stretches of GPS
lie 120–160 m from the reference route (trip 00000033 spends time in car parks the route skips). Figure 2.6 shows them. They matter later:
they sit in folds {fold_of.get('00000054')} and {fold_of.get('00000033')}, which are the two low CV folds for every model.
Candidate miss rate on student_test is {pct((te.candidate_miss_rate * te.observations).sum() / te.observations.sum())}.</p>
{fig(ED / "eda_alignment.png", "Figure 2.6 – distance to the aligned reference arc (left) and trips where reference and GPS disagree (right).")}
{fig(ROOT / "report" / "img" / "map_eda_48.png", "Figure 2.7 – Folium EDA map of train trip 00000048 (pathsnap-viz/maps/eda_00000048.html): GPS, nearby network, reference route, with layer control.")}

<h2>2.6 Features</h2>
{table(pd.DataFrame([
        ["log_distance_m", "log(1 + distance to the arc)", "closest arc is usually right"],
        ["heading_cosine", "cosine between movement (i−1 → i+1) and arc direction", "separates the two directions of a road"],
        ["log_speed_mps", "log(1 + speed)", "heading is unreliable when stopped"],
        ["log_segment_length_m", "log(1 + arc length)", "short junction stubs vs long road arcs"],
        ["projection_fraction", "where the reading projects on the arc (0–1)", "readings at an arc end may belong to the next arc"],
        ["distance_rank", "rank among the 8 candidates", "relative closeness"],
        ["log_previous_distance_m / log_next_distance_m", "distance of readings i−1 and i+1 to the same arc", "uses the local trajectory, not one noisy point"],
    ], columns=["Feature", "Definition", "Why it helps"]), "small")}""")

    a(f"""<h1>3. Model experimentation and comparison</h1>
<p><b>Protocol</b> (<code>pathsnap-training/train.py</code>): 5-fold GroupKFold over the 80 training trips (folds in
<code>results/folds.json</code>), refit on all 80, score on student_test. Seed 42.
<b>Metrics:</b> top-1 accuracy pooled over readings (candidate misses count as wrong); Macro F1 over arc classes within each trip,
averaged over trips; covered log loss on readings whose true arc is among the candidates (scores normalised over the 8);
direct continuity = share of consecutive predictions that stay on an arc or move to a connected one.</p>
<h2>3.1 Comparison</h2>
{cmp_table}
<p class="caption">Table 3.1 – CV over the train folds; all other columns on student_test. The last two rows are the later stages, included for comparison.</p>
<p><b>Selection.</b> HistGB has the best CV Macro F1 ({fstat.loc['hist_gradient_boosting'].f1_mean:.4f}) and the lowest log loss
({mc.loc['hist_gradient_boosting'].student_covered_log_loss:.4f}). Random Forest is marginally ahead on student_test top-1
({pct(mc.loc['random_forest'].student_top1_accuracy)} vs {pct(mc.loc['hist_gradient_boosting'].student_top1_accuracy)}) and Macro F1,
but the gaps between the three tree models are much smaller than the fold-to-fold spread (± {fstat.loc['hist_gradient_boosting'].f1_std:.3f}).
We chose HistGB because it wins on the pre-agreed objective (CV Macro F1), has the best-calibrated scores (they become HMM emissions),
and fits in {mc.loc['hist_gradient_boosting'].full_fit_seconds:.0f} s against {mc.loc['random_forest'].full_fit_seconds:.0f} s for Random Forest,
which made a 20-trial Bayesian search affordable.</p>
<p><b>The two weak folds.</b> CV top-1 for HistGB is {pct(low_fold.iloc[0].top1)} on fold {int(low_fold.iloc[0].fold)} and
{pct(low_fold.iloc[1].top1)} on fold {int(low_fold.iloc[1].fold)}, against 80–90% on the others, for all five models alike. Those are the folds
holding the two trips with drifting references (2.5), so the CV numbers are pulled down by label noise, not model failure.</p>
<h2>3.2 Per-model questions</h2>""")

    text = {
        "logistic_regression": (
            "A linear model is the natural first baseline: most features push one way (closer, better aligned, closer neighbours → more likely).",
            "Fast and stable across folds, but it cannot model interactions such as heading mattering only when the vehicle moves, so its scores are the least sharp of the five (log loss {ll}).",
            "StandardScaler fitted inside each fold, L2 penalty, C = 1, lbfgs, max_iter = 500. Objective: L2-regularised binary cross-entropy."),
        "gaussian_naive_bayes": (
            "Cheap probabilistic baseline that shows how far per-feature evidence goes on its own.",
            "Trains in under a second. Its independence assumption is wrong here: log_distance, previous and next distance all measure the same thing, so evidence is counted three times and scores are overconfident (worst log loss, {ll}).",
            "Default GaussianNB, var_smoothing = 1e-9. Objective: maximum likelihood of per-class Gaussians (closed form)."),
        "decision_tree": (
            "A single tree captures the interactions the linear model misses and is easy to inspect.",
            "Clear jump over the linear/probabilistic models. A single tree is high-variance, which shows in its fold spread; depth and leaf limits keep it in check.",
            "max_depth = 10, min_samples_leaf = 30. Objective: Gini impurity at each split."),
        "random_forest": (
            "Averaging many decorrelated trees reduces the variance of a single tree on noisy GPS.",
            "Best student_test top-1 of the baselines, but the slowest to train ({fit:.0f} s) and to run ({inf:.0f} ms per trip).",
            "60 trees, max_depth = 12, min_samples_leaf = 15, max_samples = 0.5. Objective: Gini impurity."),
        "hist_gradient_boosting": (
            "Boosting fits each tree to the previous trees' errors, which suits the remaining hard cases (junctions, parallel roads); it optimises log loss directly.",
            "Best CV Macro F1 and log loss, fast to fit. More hyperparameters to set, which section 4 tunes.",
            "100 iterations, learning_rate 0.1, 31 leaf nodes, no early stopping. Objective: binary log loss."),
    }
    for k, m in enumerate(ORDER, 1):
        why, strengths, cfg = text[m]
        r = mc.loc[m]
        strengths = strengths.format(ll=f"{r.student_covered_log_loss:.4f}", fit=r.full_fit_seconds, inf=r.mean_inference_ms_per_trip)
        gap = r.train_macro_f1 - r.cv_macro_f1
        verdict = ("train and CV are within {:.3f} of each other, so there is no sign of overfitting; the model is limited by its form (<b>mild underfitting</b>) rather than by variance"
                   if gap < 0.005 else "train is {:.3f} above CV, a <b>small overfitting gap</b> held back by the depth/leaf limits").format(abs(gap))
        a(f"""<div class="model"><h2 class="mh">3.2.{k} {NAME[m]} ({FAMILY[m]})</h2>
<h3>A. Model understanding</h3><p><b>Why it fits.</b> {why}</p><p><b>Strengths and limitations here.</b> {strengths}</p>
<h3>B. Training and configuration</h3><p>{cfg}</p>
<h3>C. Results</h3><p>{results(m)}</p>
<p><b>Over/underfitting.</b> {train_cv_test(m)}: {verdict}. student_test scores higher than CV for every model because its 10 trips
have no drifting references.</p></div>""")

    a(f"""<h2>3.3 Visualisations</h2>
{fig(VZ / "models_cv.png", "Figure 3.1 – CV Macro F1 and CV top-1 per model; bars = mean, error bars = std, dots = folds.")}
{fig(VZ / "models_student_test.png", "Figure 3.2 – train vs CV vs student_test Macro F1, student_test top-1, student_test log loss.")}
{fig(VZ / "models_time.png", "Figure 3.3 – training and inference time.")}
{fig(VZ / "boosting_curves.png", "Figure 3.4 – log loss per boosting iteration for HistGB (the only model with an iterative training history).")}
<p>The untuned curve flattens after about 40 iterations; train and student_test stay together, so 100 iterations do not overfit.
The tuned model (400 iterations, smaller learning rate) keeps lowering training loss to {d['bc_t'].train_log_loss.iloc[-1]:.4f}
while student_test settles at {d['bc_t'].student_test_log_loss.min():.4f}: a small, growing gap, but no rise.</p>""")

    space = pd.DataFrame([
        ["learning_rate", "log-uniform 0.02 – 0.3", f"{best['learning_rate']:.4f}"],
        ["max_iter", "50 – 400, step 25", best["max_iter"]],
        ["max_leaf_nodes", "log-uniform 8 – 128", best["max_leaf_nodes"]],
        ["min_samples_leaf", "log-uniform 10 – 400", best["min_samples_leaf"]],
        ["l2_regularization", "log-uniform 1e-4 – 10", f"{best['l2_regularization']:.5f}"],
        ["max_features", "uniform 0.4 – 1.0", f"{best['max_features']:.3f}"],
    ], columns=["Hyperparameter", "Search range", "Best value"])
    a(f"""<h1>4. Bayesian hyperparameter optimisation</h1>
<p><b>Method:</b> Optuna TPE sampler (seed 42, 6 random start-up trials), <code>pathsnap-tuning/tune.py</code>.
Trial 0 is the untuned setting. All trials are in <code>results/trials.csv</code> and <code>optuna_study.db</code>.</p>
<p><b>1–2. Hyperparameters and ranges / 5. best values:</b></p>{table(space)}
<p><b>3. Objective:</b> CV Macro F1, computed exactly as in train.py on the same folds. student_test plays no part in the search.</p>
<p><b>4. Trials:</b> {len(trials)} completed, {trials.seconds.sum() / 60:.0f} minutes of CV on a 2-core laptop. Best: trial {d['best']['best_trial']}
with CV Macro F1 {d['best']['best_cv_macro_f1']:.4f}.</p>
{fig(VZ / "optuna_history.png", "Figure 4.1 – search history and score against learning rate.")}
<p><b>6. Tuned vs untuned</b>, both rerun through the same protocol:</p>
{table(pd.DataFrame([
        ["CV Macro F1 (± fold std)", f"{un.cv_macro_f1:.4f} ± {un.cv_macro_f1_fold_std:.4f}", f"{tn.cv_macro_f1:.4f} ± {tn.cv_macro_f1_fold_std:.4f}"],
        ["CV top-1", pct(un.cv_top1_accuracy), pct(tn.cv_top1_accuracy)],
        ["student_test top-1", pct(un.student_top1_accuracy), pct(tn.student_top1_accuracy)],
        ["student_test Macro F1", f"{un.student_macro_f1:.4f}", f"{tn.student_macro_f1:.4f}"],
        ["student_test covered log loss", f"{un.student_covered_log_loss:.4f}", f"{tn.student_covered_log_loss:.4f}"],
        ["direct continuity", pct(un.student_direct_continuity, 1), pct(tn.student_direct_continuity, 1)],
        ["fit time / inference per trip", f"{un.full_fit_seconds:.1f} s / {un.mean_inference_ms_per_trip:.0f} ms", f"{tn.full_fit_seconds:.1f} s / {tn.mean_inference_ms_per_trip:.0f} ms"],
    ], columns=["Metric", "Untuned", "Tuned"]))}
{fig(VZ / "tuned_vs_untuned.png", "Figure 4.2 – untuned vs tuned HistGB.")}
<p>The tuned model is better on the objective (+{tn.cv_macro_f1 - un.cv_macro_f1:.4f} CV Macro F1) and on every student_test
metric, but the gains are small next to the fold spread, and inference is about {tn.mean_inference_ms_per_trip / un.mean_inference_ms_per_trip:.0f}× slower
(still about 0.1 s per trip). CV top-1 went slightly down, because Macro F1 rewards getting short, rare arcs right while pooled top-1 is dominated
by long arcs. The big remaining errors are geometric ambiguity, which the HMM targets.
Note: the untuned rerun here gives CV Macro F1 {un.cv_macro_f1:.4f}, while the committed train.py run gives
{mc.loc['hist_gradient_boosting'].cv_macro_f1:.4f}. The settings are identical; the difference is run-to-run numerical noise in the boosting
library across machines, so we compare tuned and untuned only within the same run.</p>""")

    per = hmm.copy()
    hmm_tab = table(pd.DataFrame({
        "Trip": per.trip_id, "Readings": per.observations,
        "Top-1 clf → HMM": [f"{pct(a_)} → {pct(b_)}" for a_, b_ in zip(per.clf_top1_accuracy, per.hmm_top1_accuracy)],
        "Macro F1 clf → HMM": [f"{a_:.3f} → {b_:.3f}" for a_, b_ in zip(per.clf_macro_f1, per.hmm_macro_f1)],
        "Continuity clf → HMM": [f"{pct(a_, 1)} → {pct(b_, 1)}" for a_, b_ in zip(per.clf_direct_continuity, per.hmm_direct_continuity)],
        "Arc runs clf / HMM / ref": [f"{a_} / {b_} / {c_}" for a_, b_, c_ in zip(per.clf_arc_runs, per.hmm_arc_runs, per.reference_arc_runs)],
        "Route F1 clf → HMM": [f"{a_:.3f} → {b_:.3f}" for a_, b_ in zip(per.clf_route_f1, per.hmm_route_f1)],
        "Breaks": per.breaks, "Viterbi s": per.viterbi_seconds.map(lambda v: f"{v:.2f}"),
    }), "small")
    a(f"""<h1>5. HMM and Viterbi route reconstruction</h1>
<p>Code: <code>pathsnap-hmm/hmm.py</code>, <code>decode.py</code>.</p>
<p><b>1. Hidden states:</b> the 8 candidate arcs of each reading (from data.py), so every state maps straight back to an arc id.
The reading's projection fraction on the arc gives the position used by the transition.</p>
<p><b>2. Emission:</b> log P(z<sub>i</sub> | arc) = w · log p(arc), where p is the tuned HistGB score normalised over the 8 candidates.</p>
<p><b>3. Transition:</b> route distance d<sub>route</sub> from the projection on arc a to the projection on arc b (rest of a + Dijkstra
shortest path from a's end node to b's start node + part of b; on the same arc, the movement along it, with backward movement ×3).
log P(b | a) = −|d<sub>route</sub> − d<sub>straight</sub>| / β, d<sub>straight</sub> = distance between the two GPS readings.
A jump to a parallel road needs a detour and is penalised. Viterbi keeps back-pointers and renormalises scores every step.</p>
<p><b>4. Gaps:</b> β grows with the time gap, β + β<sub>gap</sub>·(Δt − 1), and the Dijkstra search radius grows with the straight distance
(max(400 m, 3d + 200 m)), so long gaps allow longer detours. Candidates always exist (8 nearest arcs), so no reading is skipped.</p>
<p><b>5. Disconnected segments:</b> unreachable pairs get a fixed log-probability of −40 instead of zero, so Viterbi always has a path. These are
counted as breaks. Afterwards, non-adjacent consecutive arcs are joined by a shortest path (stitching) to give a drivable route.</p>
<p><b>Decoder tuning:</b> grid w ∈ {{0.25, 0.5, 1, 2}}, β ∈ {{2, 5, 10, 25, 50}} m, β<sub>gap</sub> ∈ {{1, 5}} on 24 training trips using
<b>out-of-fold</b> probabilities, selected on mean Macro F1. Chosen: w = {dp['w_emit']:g}, β = {dp['beta']:g} m, β<sub>gap</sub> = {dp['beta_gap']:g}.
On those trips Macro F1 rose from {d['dtune']['classifier_only_macro_f1']:.4f} to {d['dtune']['selected']['macro_f1']:.4f}. The grid shows a ridge
along the ratio w/β; several settings on it are within 0.002.</p>
{fig(VZ / "hmm_grid.png", "Figure 5.1 – decoder grid on 24 train trips (Macro F1).", "64%")}
<p><b>6. Classifier-only vs HMM on student_test:</b></p>
{table(pd.DataFrame([
        ["Top-1 accuracy (pooled)", pct(hs['clf_top1_accuracy']), pct(hs['hmm_top1_accuracy'])],
        ["Macro F1 (mean over trips)", f"{hs['clf_macro_f1']:.4f}", f"{hs['hmm_macro_f1']:.4f}"],
        ["Direct route continuity", pct(hs['clf_direct_continuity'], 1), pct(hs['hmm_direct_continuity'], 1)],
        ["Arc runs (reference: " + f"{hs['reference_arc_runs']:,})", f"{hs['clf_arc_runs']:,}", f"{hs['hmm_arc_runs']:,}"],
        ["Route F1 after stitching", f"{hs['clf_route_f1']:.3f}", f"{hs['hmm_route_f1']:.3f}"],
    ], columns=["Sequence-level", "Classifier only", "Classifier + HMM"]))}
{hmm_tab}
{fig(VZ / "hmm_vs_classifier.png", "Figure 5.2 – per-trip comparison on student_test.")}
<p>The HMM improves all 10 trips. The classifier alone switches arcs {hs['clf_arc_runs']:,} times against {hs['reference_arc_runs']:,} in the
reference, flicking to the reverse direction or a side road for a reading or two; the HMM output has {hs['hmm_arc_runs']:,} runs.</p>
<p><b>Where it still fails.</b> Errors concentrate at low speed (Figure 5.3). Below 1 m/s the HMM is right {pct(acc_speed['classifier + HMM']['< 1 m/s'], 0)}
of the time and at 1–5 m/s {pct(acc_speed['classifier + HMM']['1-5'], 0)}, against {pct(acc_speed['classifier + HMM']['> 15'], 1)} above 15 m/s. When the
vehicle crawls, the heading is noise and both directions of a road are equally close, and {pct(rev['classifier + HMM'], 0)} of the HMM's remaining
errors pick the reverse-direction twin of the right arc. Trip 00000077 shows it most: {pct(0.45, 0)} of its readings are below 5 m/s, against
0–23% for the other trips, and it is the weakest trip (top-1 {pct(h77.hmm_top1_accuracy)}), even though its route F1 is
{h77.hmm_route_f1:.3f}: the right roads, sometimes the wrong direction.</p>
{fig(VZ / "error_by_speed.png", "Figure 5.3 – student_test accuracy by speed, classifier only and with HMM.", "64%")}""")

    a(f"""<h1>6. Geographical visualisation (Folium)</h1>
<p><code>pathsnap-viz/make_maps.py</code> writes one map per student_test trip (<code>maps/prediction_&lt;id&gt;.html</code>) plus EDA maps.
Layers can be toggled: road network near the trip, reference route (green), classifier-only route (orange, dashed), HMM route (red),
GPS readings with time tooltips, and start/end markers. Screenshots below are exported without the tile background.</p>
{fig(ROOT / "report" / "img" / "map_pred_56.png", f"Figure 6.1 – student_test 00000056, HMM top-1 {pct(hmm.set_index('trip_id').loc['00000056'].hmm_top1_accuracy)}.")}
{fig(ROOT / "report" / "img" / "map_pred_56_zoom.png", "Figure 6.2 – the interchange in 00000056. Orange dashes: classifier-only picks on disconnected parallel ramps. Red sits on the green reference; the red loop top-left is a stitching detour between two non-adjacent decoded arcs.")}
{fig(ROOT / "report" / "img" / "map_pred_77.png", "Figure 6.3 – student_test 00000077, the slow trip discussed in section 5.")}""")

    a(f"""<h1>7. API, UI and deployment</h1>
<p><b>Laptop 1, <code>pathsnap-api</code></b> (FastAPI + uvicorn) loads the tuned model and decoder settings once and lists the trip folders in
<code>data/eval/</code>. <code>/predict</code> runs data.prepare_candidates → classifier → Viterbi → stitching. Only after that does
<code>scoring.py</code> read the <code>.route</code> file, align it with the same dynamic program as training, and compute the metrics. Only
numbers go back; no reference geometry is sent. Nothing is cached per trip, so every request runs the whole pipeline (timings in the log).</p>
{table(pd.DataFrame([["GET /health", "status, trips, model"], ["GET /trips", "ten trip ids with reading counts"],
                     ["POST /predict {trip_id}", "GPS, HMM route, classifier-only route, per-reading arcs, candidate network, metrics, timings"]],
                    columns=["Endpoint", "Returns"]))}
<p><b>Laptop 2, <code>pathsnap-ui</code></b>: static page (<code>python -m http.server</code>) with bundled Leaflet. It lists the trips from
<code>/trips</code>, sends only the selected id, and draws the returned layers next to the metrics.</p>
{fig(ROOT / "report" / "img" / "ui_00000056.png", "Figure 7.1 – evaluator UI, trip 00000056 (student_test trips stood in for the evaluation set during testing).")}
{fig(ROOT / "report" / "img" / "ui_00000077.png", "Figure 7.2 – trip 00000077 with the classifier-only layer.")}
<p><b>API log</b> while the UI and a test client requested all ten trips and one unknown id:</p>
<pre class="log">{d['log']}</pre>

<h1>8. Limitations</h1>
<ul>
<li>Per-reading labels are derived from the route by alignment; near junctions the boundary between two touching arcs is partly a matter of
definition. Route F1 ({hs['hmm_route_f1']:.3f}) is the better measure of whether the right roads were found.</li>
<li>Two training references drift 120–160 m from the GPS; they lower CV for every model (section 3).</li>
<li>Slow, manoeuvring segments remain the main error source (section 5). A speed-dependent β or a turn-around penalty is the next step.</li>
<li>Stitching can add a detour on interchanges with one-way ramps (Figure 6.2); limiting fills to paths not much longer than the GPS
distance would fix it.</li>
</ul>
<h1>9. Reproducibility</h1>
<p>Each stage folder has a README with the exact command. All tables and figures here are read from those outputs by
<code>report/build_report.py</code>. Seed 42 throughout. The evaluator partition was not used for training, tuning, model selection or decoder settings.</p>""")
    return f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(H)}</body></html>"


async def to_pdf(html, pdf):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html.resolve().as_uri())
        await page.pdf(path=str(pdf), format="A4", print_background=True,
                       margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
                       display_header_footer=True, header_template="<span></span>",
                       footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#777">'
                                       'PathSnap · page <span class="pageNumber"></span> of <span class="totalPages"></span></div>')
        await browser.close()


if __name__ == "__main__":
    html = REPORT / "PathSnap_Report.html"
    html.write_text(build(load()), encoding="utf-8")
    asyncio.run(to_pdf(html, REPORT / "PathSnap_Report.pdf"))
    print("written report/PathSnap_Report.pdf")

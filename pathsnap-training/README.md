# PathSnap model experiments

This folder contains the five classifier baselines and their actual experiment outputs. Bayesian tuning, HMM/Viterbi, Folium maps, and the two-laptop deployment are separate stages.

## Reproduce

From this folder, using Python 3.14 and the original student dataset ZIP:

```powershell
python -m pip install -r requirements.txt
python data.py --zip "$env:USERPROFILE\Downloads\Student folder.zip" --out cache
python train.py --cache cache --out results
python verify.py --cache cache --results results
python plot_results.py results/model_comparison.csv
```

Use a fresh cache for a different archive. The raw data and feature cache are excluded from this bundle. The GitHub video was located, but GitHub would not preview it at its size. The run used the written guidelines and inspected archive.

## Data and labels

The student archive contains 80 training trips with 208,417 GPS readings and 10 student-test trips with 19,160 readings. No evaluator partition was used. Road segments have trip-local row IDs. Each estimator learns a binary match score for a candidate segment from its features, allowing the same model to score segments on unseen trips.

Candidate generation finds the eight nearest finite segments in a local equirectangular projection, breaking distance ties by arc ID. Eight is an experiment setting. Emission inputs include distance, direction agreement, estimated speed, segment length, position along the segment, candidate rank, and distances from adjacent observations. Candidate generation and features do not use the reference route.

The route file contains an ordered segment sequence. For supervision and scoring, a separate dynamic program assigns GPS readings to nondecreasing positions in that sequence, minimizing summed point-to-segment distance. This creates derived point labels from the supplied route. Label ambiguity at junctions, GPS drift, and errors in the supplied route can affect all reported scores. The label aligner is distinct from the future prediction-time HMM.

## Evaluation

- Five folds hold out complete training trips. The final baseline weights fit all 80 training trips and are evaluated on the same 10 student-test trips. Logistic regression scaling is fit independently inside each fold.
- Top-1 accuracy is pooled over student-test GPS observations. Candidate misses count as incorrect. Macro F1 averages arc classes within each trip and then averages trips equally. Arc IDs are never treated as shared classes across trips.
- Log loss covers observations with a reference arc among the eight candidates. Candidate scores are normalized within each candidate set. Coverage is reported separately. Candidate misses are excluded from supervised training rows.
- Direct route continuity is the fraction of adjacent predictions that stay on the same arc or move onto a directly connected directed arc. A legitimate move over intermediate arcs may fail this check. Inference time measures estimator scoring and ranking on cached features.
- Each model folder contains its settings, a fitted estimator, per-trip scores, held-out fold scores, predictions, and sparse arc confusion counts.

Histogram Gradient Boosting is the leading baseline for a tuning stage that uses cross-validation Macro F1 as its objective. Random Forest is marginally ahead on student-test accuracy and Macro F1, so preserve both baseline results when discussing model selection.

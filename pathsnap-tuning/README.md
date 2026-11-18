# Bayesian hyperparameter optimisation (HistGradientBoosting)

`tune.py` tunes the Histogram Gradient Boosting baseline from `pathsnap-training`
with Optuna's TPE sampler.

- **Objective:** cross-validation Macro F1, computed exactly as in `train.py`,
  on the same trip folds (`pathsnap-training/results/folds.json`).
- **Trial 0** starts near the untuned `train.py` setting: its L2 regularisation is
  `1e-4` (the search's lower bound), whereas the untuned baseline uses `0`.
  The untuned comparison is fitted separately with the actual `train.py` defaults.
- **student_test** is not used in the search. It is only used afterwards to compare
  the untuned and tuned models.
- **Out-of-fold probabilities** of the tuned model are saved for tuning the HMM decoder.

```bash
cd pathsnap-tuning
python -m pip install -r requirements.txt
python tune.py --cache ../pathsnap-training/cache --trials 20
```

Outputs in `results/`:

| file | contents |
|---|---|
| `trials.csv`, `optuna_study.db` | every trial with its parameters, CV Macro F1, CV top-1 and run time |
| `best_params.json` | best parameters and score |
| `tuned_vs_untuned.csv` | both models through the same CV protocol and on student_test |
| `tuned_model/` | tuned model, its CV and student_test scores, predictions, out-of-fold and student_test probabilities |

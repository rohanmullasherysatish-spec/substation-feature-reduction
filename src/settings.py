"""Shared paths, allocation settings and declared modelling choices."""

from pathlib import Path

# Resolve the root once; protocol packages import this same location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allocate whole measurement groups; fix seeds and fold choice before modelling.
OUTER_FOLDS = 5
TEST_FOLD = 0
OUTER_SEED = 42
VALIDATION_FOLDS = 3
VALIDATION_SEED = 43
# Accept a test fraction within five percentage points of the 20% target.
MAX_TEST_FRACTION_DEVIATION = 0.05

# Declare the comparison panel and candidate order before fitting.
# Four classifiers use trees, so shared importance describes this chosen panel.
CLASSIFIER_PANEL = (
    "logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm",
)
REFERENCE_STRATEGY = "most_frequent"
# Only continuous/count values are standardised; one-hot columns remain binary.
STANDARDIZE_NUMERIC_FOR = ("logistic_regression",)

# One estimator seed, one fit at a time and bounded numerical-library threads.
ESTIMATOR_SEED = 44
TRAINING_WORKERS = 2
PREDICTION_BATCH_ROWS = 4096
# A solver warning triggers a larger ceiling; converged fits are not repeated.
LR_ITERATION_CAPS = (5000, 10000, 20000)
SELECTION_TIE_TOLERANCE = 1e-6

# Feature selection uses only each fold's training rows, then all development.
# Equal true-class and output-class weights keep the summary aligned with macro F1.
# These are one declared aggregation policy, not extra experimental conditions.
SHAP_FOREGROUND_PER_CLASS = 128
SHAP_BACKGROUND_PER_CLASS = 16
SELECTION_SAMPLE_SEED = 45
SHAP_ADDITIVITY_ATOL = 1e-5
SHAP_ADDITIVITY_RTOL = 1e-5
# MI is a separate capped training-sample benchmark; no rare rows are duplicated.
MI_ROWS_PER_CLASS = 2000
MI_NEIGHBOURS = 3
MI_SAMPLE_SEED = 46
MI_ESTIMATION_SEED = 47
# Count original fields after training-only screening; keep every encoded component.
RETENTION_FRACTIONS = (0.25, 0.50)
REDUCED_PARAMETER_POLICY = "reuse_selected_full_parameters"
PREDICTION_TIMING_REPEATS = 3
# Conditional uncertainty resamples whole saved groups, paired across conditions.
# Two groups permit non-degenerate resampling; small supports remain imprecise.
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 48
MIN_BOOTSTRAP_GROUPS_PER_CLASS = 2

# Balanced sample weights are supplied once by the training helper, not here.
# These fixed parameters are merged with one candidate below for each fit.
MODEL_FIXED_PARAMETERS = {
    "logistic_regression": {"solver": "lbfgs", "l1_ratio": 0.0,
                            "max_iter": 5000, "tol": 1e-4},
    "decision_tree": {},
    "random_forest": {"n_estimators": 200, "max_features": "sqrt", "bootstrap": True},
    "xgboost": {"n_estimators": 200, "objective": "multi:softprob", "tree_method": "hist",
                "device": "cpu", "eval_metric": "mlogloss", "subsample": 1.0,
                "colsample_bytree": 1.0, "verbosity": 1},
    "lightgbm": {"n_estimators": 200, "objective": "multiclass", "learning_rate": 0.05,
                 "subsample": 1.0, "colsample_bytree": 1.0, "deterministic": True,
                 "force_col_wise": True, "verbosity": -1},
}
# Four candidates are a bounded search, not equal compute or a claim of optimality.
# When mean validation macro F1 differs by at most the tolerance, use listed order.
MODEL_CANDIDATES = {
    "logistic_regression": tuple({"C": value} for value in (0.01, 0.1, 1.0, 10.0)),
    "decision_tree": tuple({"max_depth": depth, "min_samples_leaf": leaf}
                           for depth, leaf in ((8, 10), (8, 2), (16, 10), (16, 2))),
    "random_forest": tuple({"max_depth": depth, "min_samples_leaf": leaf}
                           for depth, leaf in ((12, 5), (12, 1), (24, 5), (24, 1))),
    "xgboost": tuple({"max_depth": depth, "learning_rate": rate}
                     for depth, rate in ((3, 0.05), (3, 0.1), (6, 0.05), (6, 0.1))),
    "lightgbm": tuple({"num_leaves": leaves, "min_child_samples": child}
                      for leaves, child in ((15, 20), (15, 5), (31, 20), (31, 5))),
}


def protocol_settings(protocol):
    """Choose an explicit protocol schema without mixing the two input spaces."""
    if protocol == "iec104":
        from src.iec104 import settings as config
    elif protocol == "iec61850":
        from src.iec61850 import settings as config
    else:
        raise ValueError("Unknown protocol.")
    return config

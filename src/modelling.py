"""Classifier fitting, aligned probabilities and reusable full-input pipelines."""

import importlib.metadata
import time
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src import settings
from src.checks import require


class TrainingFitError(ValueError):
    """Keep failed fitting costs and warnings available without treating a fit as usable."""

    def __init__(self, message, fit_info):
        super().__init__(message)
        self.fit_info = fit_info


def make_estimator(name, parameters, class_count):
    """Construct one declared estimator; weighting is applied only during fitting."""
    require(name in settings.CLASSIFIER_PANEL or name == "dummy", "Unknown classifier.")
    parameters = dict(parameters)
    class_count = int(class_count)
    require(class_count >= 2, "Multiclass fitting requires at least two declared classes.")
    require(parameters.get("class_weight") is None,
            "Do not combine class weights with the declared balanced observation weights.")
    if name == "dummy":
        require(parameters.get("strategy", settings.REFERENCE_STRATEGY) == settings.REFERENCE_STRATEGY,
                "Reference strategy differs from settings.")
        return DummyClassifier(**(parameters | {"strategy": settings.REFERENCE_STRATEGY}))
    parameters["random_state"] = settings.ESTIMATOR_SEED
    if name in {"random_forest", "xgboost", "lightgbm"}:
        parameters["n_jobs"] = settings.TRAINING_WORKERS
    if name in {"xgboost", "lightgbm"}:
        parameters["num_class"] = class_count
        try:
            if name == "xgboost":
                from xgboost import XGBClassifier
                return XGBClassifier(**parameters)
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**parameters)
        except Exception as error:
            raise ValueError(f"Cannot load {name}'s native backend: {error}") from error
    return {"logistic_regression": LogisticRegression, "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier}[name](**parameters)


def _labels(y, rows, class_count):
    values = np.asarray(y)
    require(values.ndim == 1 and len(values) == rows and rows > 0
            and np.issubdtype(values.dtype, np.integer)
            and ((values >= 0) & (values < class_count)).all(),
            "Targets must be nonempty integer codes in the fixed class order.")
    return values


def _warning_records(caught, cap):
    counts = {}
    for item in caught:
        key = (item.category.__name__, str(item.message))
        counts[key] = counts.get(key, 0) + 1
    return [{"max_iter": cap, "category": category, "message": message, "count": count}
            for (category, message), count in counts.items()]


def fit_classifier(name, x, y, parameters, class_labels):
    """Use training-derived weights once and repair only nonconverged LR fits."""
    x = np.asarray(x)
    require(x.ndim == 2 and x.shape[1] > 0 and np.isfinite(x).all(), "Invalid training matrix.")
    classes = tuple(class_labels)
    require(len(classes) >= 2 and len(set(classes)) == len(classes), "Invalid declared class labels.")
    y = _labels(y, len(x), len(classes))
    require(set(np.unique(y)) == set(range(len(classes))), "A training class is absent.")
    weights = None if name == "dummy" else compute_sample_weight("balanced", y)
    if weights is not None:
        require(np.isfinite(weights).all() and (weights > 0).all(), "Invalid training weights.")
    caps = tuple(settings.LR_ITERATION_CAPS) if name == "logistic_regression" else (None,)
    if name == "logistic_regression":
        require(parameters.get("max_iter", caps[0]) == caps[0],
                "LR must start at the declared initial iteration cap.")
    elapsed, warning_log = 0.0, []
    for attempt, cap in enumerate(caps, start=1):
        attempted = dict(parameters)
        if cap is not None:
            attempted["max_iter"] = cap
        model = make_estimator(name, attempted, len(classes))
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=settings.TRAINING_WORKERS):
            warnings.simplefilter("always")
            started = time.perf_counter()
            if weights is None:
                model.fit(x, y)
            else:
                model.fit(x, y, sample_weight=weights)
            elapsed += time.perf_counter() - started
        warning_log.extend(_warning_records(caught, cap))
        not_converged = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        if not_converged:
            if name == "logistic_regression" and attempt < len(caps):
                print(f"Logistic Regression did not converge at {cap:,} iterations; retrying at {caps[attempt]:,}.", flush=True)
                continue
            raise TrainingFitError(
                f"{name} has unresolved convergence after {attempt} attempt(s); comparison stopped.",
                {"fit_seconds": elapsed, "fit_attempts": attempt, "effective_max_iter": cap,
                 "iterations": np.asarray(getattr(model, "n_iter_", []), dtype=int).reshape(-1).tolist(),
                 "warnings": warning_log, "training_weight_policy": "balanced_training_observation_weights"})
        require(np.array_equal(np.asarray(model.classes_), np.arange(len(classes))),
                "Fitted estimator class order differs from the target mapping.")
        iterations = np.asarray(getattr(model, "n_iter_", []), dtype=int).reshape(-1).tolist()
        return model, {"fit_seconds": elapsed, "fit_attempts": attempt,
                       "effective_max_iter": cap, "iterations": iterations,
                       "warnings": warning_log,
                       "training_weight_policy": "unweighted_most_frequent" if weights is None
                       else "balanced_training_observation_weights"}
    raise ValueError("No fitting attempt was performed.")


def _validate_probabilities(values, rows, class_count):
    require(values.shape == (rows, class_count), "Incorrect probability shape.")
    require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1 + 1e-6).all()
            and np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-5),
            "Invalid predicted probabilities.")


def probs(model, x):
    """Predict in bounded batches while preserving the declared probability order."""
    x = np.asarray(x)
    require(x.ndim == 2 and np.isfinite(x).all(), "Invalid prediction matrix.")
    classes = np.asarray(getattr(model, "classes_", []))
    class_count = len(classes)
    require(class_count >= 2 and np.array_equal(classes, np.arange(class_count)),
            "Estimator class order differs from the declared probability order.")
    output = np.empty((len(x), class_count), dtype=np.float64)
    batch_rows = settings.PREDICTION_BATCH_ROWS
    require(batch_rows > 0, "Prediction batch size must be positive.")
    with threadpool_limits(limits=settings.TRAINING_WORKERS):
        for start in range(0, len(x), batch_rows):
            block = x[start:start + batch_rows]
            values = np.asarray(model.predict_proba(block), dtype=np.float64)
            _validate_probabilities(values, len(block), class_count)
            output[start:start + len(block)] = values
    return output


def classification_measures(y, probabilities, class_labels):
    """Summarise fixed-label multiclass predictions without changing thresholds."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = tuple(class_labels)
    require(len(classes) >= 2 and len(set(classes)) == len(classes) and "normal" in classes,
            "Invalid declared class order or absent normal reference.")
    y = _labels(y, len(probabilities), len(classes))
    _validate_probabilities(probabilities, len(y), len(classes))
    predicted = probabilities.argmax(axis=1)
    labels = np.arange(len(classes))
    matrix = confusion_matrix(y, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=labels, zero_division=0)
    normal = classes.index("normal")
    normal_support = int(support[normal])
    false_alarms = normal_support - int(matrix[normal, normal])
    overall = {"accuracy": float(np.mean(y == predicted)), "macro_f1": float(f1.mean()),
               "weighted_f1": float(np.average(f1, weights=support)),
               "normal_false_alarm_rate": false_alarms / normal_support if normal_support else None,
               "errors": int(np.count_nonzero(y != predicted)), "records": len(y)}
    per_class = [{"class_label": label, "precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i]),
                  "errors": int(support[i] - matrix[i, i])}
                 for i, label in enumerate(classes)]
    return overall, per_class


class FullModel:
    """Keep a classifier with the exact preprocessing and numeric view it used."""

    def __init__(self, processor, estimator, model_name, parameters, training_scope, class_labels):
        self.preprocessor = processor
        self.estimator = estimator
        self.model_name = model_name
        self.parameters = dict(parameters)
        self.training_scope = training_scope
        self.classes_ = tuple(class_labels)
        require(np.array_equal(estimator.classes_, np.arange(len(self.classes_))),
                "Pipeline class labels differ from its fitted estimator.")

    def predict_proba(self, frame):
        self.preprocessor._validate_frame(frame)
        require(np.array_equal(self.estimator.classes_, np.arange(len(self.classes_))),
                "Saved estimator and pipeline classes differ.")
        output = np.empty((len(frame), len(self.classes_)), dtype=np.float64)
        batch_rows = settings.PREDICTION_BATCH_ROWS
        require(batch_rows > 0, "Prediction batch size must be positive.")
        for start in range(0, len(frame), batch_rows):
            block = frame.iloc[start:start + batch_rows]
            encoded = self.preprocessor.transform_for_model(block, self.model_name)
            output[start:start + len(block)] = probs(self.estimator, encoded)
        return output

    def predict(self, frame):
        return np.asarray(self.classes_)[self.predict_proba(frame).argmax(axis=1)]


def check_backends(class_count):
    """Import native backends and construct declared estimators without fitting."""
    for name in settings.CLASSIFIER_PANEL:
        fixed = settings.MODEL_FIXED_PARAMETERS[name]
        candidate = settings.MODEL_CANDIDATES[name][0]
        make_estimator(name, fixed | candidate, class_count)
    make_estimator("dummy", {}, class_count)
    try:
        return {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm",
                 "pyarrow", "joblib", "threadpoolctl")}
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f"Required training package is unavailable: {error}") from error

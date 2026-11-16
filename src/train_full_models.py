"""Tune full-input classifiers using a protocol's saved grouped development folds.

Save selected fold models for explanations and fit the selected recipes on all
development rows. No test predictions or feature selection occur in this command.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src import settings
from src.modelling import (
    FullModel, TrainingFitError, check_backends, classification_measures, fit_classifier, probs,
)
from src.checks import require, invalidate_results


def protocol_classes(protocol):
    return tuple(settings.protocol_settings(protocol).CLASSES)


def training_recipe(protocol="iec104"):
    """Record the declared search and interpretation, rather than only its winner."""
    config = settings.protocol_settings(protocol)
    recipe = {
        "classifier_panel": list(settings.CLASSIFIER_PANEL),
        "reference_strategy": settings.REFERENCE_STRATEGY,
        "fixed_parameters": settings.MODEL_FIXED_PARAMETERS,
        "candidate_grid": {name: list(values) for name, values in settings.MODEL_CANDIDATES.items()},
        "estimator_seed": settings.ESTIMATOR_SEED,
        "workers": settings.TRAINING_WORKERS,
        "parallel_fits": 1,
        "class_weighting": "balanced training-observation weights once per learned classifier; dummy unweighted",
        "selection_metric": "mean of three validation macro F1 scores, fixed eight-class label order",
        "tie_tolerance": settings.SELECTION_TIE_TOLERANCE,
        "tie_break": "first candidate in declared order within tolerance of the maximum",
        "lr_iteration_caps": list(settings.LR_ITERATION_CAPS),
        "lr_retry_rule": "retry only warning fits; stop on unresolved convergence",
        "validation_interpretation": "selection evidence, not an independent estimate of the entire tuning procedure",
        "test_scoring": False,
        "grouping_fields": list(config.GROUPING_FIELDS) if protocol == "iec104" else [],
        "grouping_dtype": config.GROUPING_DTYPE,
        "outer_seed": settings.OUTER_SEED,
        "validation_seed": settings.VALIDATION_SEED,
        "classes": list(config.CLASSES),
        "standardize_numeric_for": list(settings.STANDARDIZE_NUMERIC_FOR),
        "prediction_timing": "estimator predict_proba on already encoded inputs; excludes preprocessing and flow extraction",
        "transform_timing": "one shared transformation per model/scope; repeated table value must not be summed across candidates",
    }
    classes = protocol_classes(protocol)
    if protocol == "iec61850":
        fields = json.loads((settings.PROJECT_ROOT/"data/iec61850/features.json").read_text())
        recipe.update(classes=list(classes), grouping_fields=fields["numeric_grouping_fields"],
                      predictor_fields=fields["predictor_fields"],
                      preprocessing_policy=config.preprocessing_policy(),
                      selection_metric="mean of three validation macro F1 scores, fixed five-class label order",
                      prediction_timing="estimator predict_proba on already encoded inputs; excludes preprocessing and packet/history extraction")
    return recipe


def load_training_inputs(root, protocol="iec104"):
    """Load the protocol's checked, training-scoped preprocessing and inputs."""
    protocol_classes(protocol)
    if protocol == "iec104":
        from src.iec104.preprocessing import load_training_inputs as load_inputs
    else:
        from src.iec61850.preprocessing import load_training_inputs as load_inputs
    return load_inputs(root)


def candidate_means(rows, model, count):
    """Require a complete three-fold comparison, then apply the fixed tie rule."""
    means = []
    for candidate in range(count):
        fits = [row for row in rows if row["model"] == model
                and row["candidate_index"] == candidate and row["scope"].startswith("fold_")]
        require(len(fits) == settings.VALIDATION_FOLDS
                and len({row["scope"] for row in fits}) == settings.VALIDATION_FOLDS,
                f"Incomplete validation comparison: {model}, candidate {candidate}")
        values = np.array([row["validation_macro_f1"] for row in fits], dtype=float)
        require(np.isfinite(values).all(), "A candidate has invalid validation scores.")
        means.append(float(values.mean()))
    maximum = max(means)
    selected = next(index for index, value in enumerate(means)
                    if maximum - value <= settings.SELECTION_TIE_TOLERANCE)
    return selected, means


def atomic_json(path, value, staging):
    """Replace one readable settings record after its complete text is written."""
    temporary = staging / path.name
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)


def save_progress(results, rows, class_rows, record, staging):
    """Keep a few consolidated outputs current, including on an interrupted run."""
    for filename, values in (("full_models.csv", rows), ("full_class_metrics.csv", class_rows)):
        temporary = staging / filename
        # An empty new run must not leave an older score table beside its settings.
        columns = ["model", "candidate_index", "scope"]
        if filename == "full_class_metrics.csv":
            columns += ["population", "class_label"]
        table = pd.DataFrame(values) if values else pd.DataFrame(columns=columns)
        table.to_csv(temporary, index=False)
        results.mkdir(parents=True, exist_ok=True)
        temporary.replace(results / filename)
    record["logical_fits_completed"] = len(rows)
    record["estimator_fit_attempts"] = sum(row["fit_attempts"] for row in rows)
    record["estimator_fit_attempts"] += record.get("failed_fit", {}).get("fit_attempts", 0)
    atomic_json(results / "selected_parameters.json", record, staging)


def score_population(model, matrix, target, class_labels):
    started = perf_counter()
    probabilities = probs(model, matrix)
    elapsed = perf_counter() - started
    overall, per_class = classification_measures(target, probabilities, class_labels)
    return overall | {"prediction_seconds": elapsed}, per_class


def fitted_pipeline(processor, model, name, parameters, scope, info, class_labels):
    used = dict(parameters)
    if name == "logistic_regression":
        used["max_iter"] = info["effective_max_iter"]
    return FullModel(processor, model, name, used, scope, class_labels)


def fit_and_measure(name, candidate, scope, parameters, processor, xtrain, ytrain,
                    xvalid, yvalid, transform_seconds, class_labels):
    """Measure each fit once; prefix training/validation metrics to keep scopes clear."""
    estimator, info = fit_classifier(name, xtrain, ytrain, parameters, class_labels)
    row = {
        "model": name, "candidate_index": candidate, "scope": scope,
        "parameters": json.dumps(parameters, sort_keys=True),
        "training_rows": len(ytrain), "validation_rows": len(yvalid),
        "active_original_fields": len(processor.active_fields_),
        "encoded_columns": processor.n_features_out_,
        "matrix_bytes": int(xtrain.nbytes + xvalid.nbytes),
        "input_transform_seconds_shared": transform_seconds,
        **{key: value for key, value in info.items() if key not in {"warnings", "iterations"}},
        "warnings": json.dumps(info["warnings"], sort_keys=True),
        "iterations": json.dumps(info["iterations"]),
    }
    class_rows = []
    for population, matrix, target in (("training", xtrain, ytrain), ("validation", xvalid, yvalid)):
        if len(target) == 0:
            continue
        try:
            overall, per_class = score_population(estimator, matrix, target, class_labels)
        except Exception as error:
            raise TrainingFitError(f"{name} prediction/metric validation failed: {error}", info) from error
        row.update({f"{population}_{key}": value for key, value in overall.items()})
        class_rows.extend({"model": name, "candidate_index": candidate, "scope": scope,
                           "population": population, **value} for value in per_class)
    pipeline = fitted_pipeline(processor, estimator, name, parameters, scope, info, class_labels)
    return pipeline, row, class_rows


def save_pipeline(pipeline, target, staging, probe):
    """Save using stable module paths and verify prediction persistence on development only."""
    temporary = staging / "selected_pipeline.joblib"
    joblib.dump(pipeline, temporary, compress=3)
    restored = joblib.load(temporary)
    expected, actual = pipeline.predict_proba(probe), restored.predict_proba(probe)
    # Parallel forests may sum identical tree outputs in a different order.
    # Bound round-off tightly and require exactly the same predicted classes.
    np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(expected.argmax(axis=1), actual.argmax(axis=1))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)


def train(root=settings.PROJECT_ROOT, overwrite=False, protocol="iec104"):
    root = Path(root).resolve()
    classes = protocol_classes(protocol)
    results = root / f"results/{protocol}/validation"
    model_names = (*settings.CLASSIFIER_PANEL, "dummy")
    require(settings.VALIDATION_FOLDS == 3, "This training recipe declares three validation folds.")
    require(set(settings.MODEL_CANDIDATES) == set(settings.CLASSIFIER_PANEL)
            and all(len(settings.MODEL_CANDIDATES[name]) == 4 for name in settings.CLASSIFIER_PANEL),
            "The declared four-candidate model panel differs.")
    scope_names = [f"fold_{fold}" for fold in range(settings.VALIDATION_FOLDS)] + ["development"]
    targets = {(name, scope): root / f"models/{protocol}/full" / name / f"{scope}.joblib"
               for name in model_names for scope in scope_names}
    protected = [results / name for name in
                 ("full_models.csv", "full_class_metrics.csv", "selected_parameters.json")] + list(targets.values())
    for path in protected:
        require(path.resolve() == path and (not path.exists() or path.is_file()),
                f"Output is redirected or is not a regular file: {path}")
    require(overwrite or not any(path.exists() for path in protected),
            "Full-training outputs already exist; deliberate replacement requires --overwrite.")
    versions = check_backends(len(classes))
    inputs, labels, scopes, processors = load_training_inputs(root, protocol)
    record = {
        "status": "running", "recipe": training_recipe(protocol), "selected": {},
        "python": platform.python_version(), "packages": versions,
        "logical_fits_completed": 0, "estimator_fit_attempts": 0,
    }
    rows, class_rows = [], []
    with TemporaryDirectory(prefix=f".train_full_{protocol}_", dir=root) as temporary:
        staging = Path(temporary)
        invalidate_results(root, protocol, 'full')
        save_progress(results, rows, class_rows, record, staging)
        try:
            for name in model_names:
                candidates = ({},) if name == "dummy" else settings.MODEL_CANDIDATES[name]
                fixed = {"strategy": settings.REFERENCE_STRATEGY} if name == "dummy" else settings.MODEL_FIXED_PARAMETERS[name]
                # Only this family's candidate models are temporary; keep its selected folds.
                with TemporaryDirectory(prefix=f"{name}_", dir=staging) as candidate_folder:
                    candidate_folder = Path(candidate_folder)
                    for scope in scope_names[:-1]:
                        mask = scopes[scope]
                        processor = processors[scope]
                        started = perf_counter()
                        xtrain = processor.transform_for_model(inputs.loc[mask], name)
                        xvalid = processor.transform_for_model(inputs.loc[~mask], name)
                        transform_seconds = perf_counter() - started
                        ytrain, yvalid = labels[mask], labels[~mask]
                        for index, candidate in enumerate(candidates):
                            parameters = fixed | candidate
                            active_fit = {"model": name, "candidate_index": index, "scope": scope}
                            print(f"{name}, {scope}, candidate {index + 1}/{len(candidates)}: "
                                  f"fitting {len(ytrain):,} training observations.", flush=True)
                            pipeline, row, per_class = fit_and_measure(
                                name, index, scope, parameters, processor, xtrain, ytrain,
                                xvalid, yvalid, transform_seconds, classes)
                            joblib.dump(pipeline, candidate_folder / f"{scope}_{index}.joblib", compress=3)
                            rows.append(row)
                            class_rows.extend(per_class)
                            save_progress(results, rows, class_rows, record, staging)
                            print(f"  validation macro F1={row['validation_macro_f1']:.6f}; "
                                  f"fit attempts={row['fit_attempts']}", flush=True)
                            del pipeline
                        del xtrain, xvalid
                    selected, means = candidate_means(rows, name, len(candidates))
                    parameters = fixed | candidates[selected]
                    selected_record = {
                        "candidate_index": selected, "parameters": parameters,
                        "candidate_mean_validation_macro_f1": means,
                        "mean_validation_macro_f1": means[selected], "pipelines": {},
                    }
                    record["selected"][name] = selected_record
                    for scope in scope_names[:-1]:
                        pipeline = joblib.load(candidate_folder / f"{scope}_{selected}.joblib")
                        probe = inputs.loc[scopes[scope]].iloc[:16]
                        target = targets[name, scope]
                        save_pipeline(pipeline, target, staging, probe)
                        selected_record["pipelines"][scope] = target.relative_to(root).as_posix()
                        del pipeline
                    save_progress(results, rows, class_rows, record, staging)
                print(f"{name}: selected candidate {selected + 1}; fitting all development.", flush=True)
                processor = processors["development"]
                started = perf_counter()
                xtrain = processor.transform_for_model(inputs, name)
                transform_seconds = perf_counter() - started
                xvalid = np.empty((0, processor.n_features_out_), dtype=np.float64)
                active_fit = {"model": name, "candidate_index": selected, "scope": "development"}
                pipeline, row, per_class = fit_and_measure(
                    name, selected, "development", parameters, processor, xtrain, labels,
                    xvalid, np.empty(0, dtype=np.int64), transform_seconds, classes)
                target = targets[name, "development"]
                save_pipeline(pipeline, target, staging, inputs.iloc[:16])
                selected_record["pipelines"]["development"] = target.relative_to(root).as_posix()
                rows.append(row)
                class_rows.extend(per_class)
                save_progress(results, rows, class_rows, record, staging)
                del pipeline, xtrain, xvalid
            expected_fits = settings.VALIDATION_FOLDS * (1 + sum(len(c) for c in settings.MODEL_CANDIDATES.values())) + len(model_names)
            expected_class_rows = (2 * settings.VALIDATION_FOLDS * (1 + sum(len(c) for c in settings.MODEL_CANDIDATES.values()))
                                   + len(model_names)) * len(classes)
            require(len(rows) == expected_fits and len(class_rows) == expected_class_rows,
                    "Unexpected logical-fit or class-metric counts; review completion before use.")
            record["status"] = "complete"
            save_progress(results, rows, class_rows, record, staging)
        except BaseException as error:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            if getattr(error, "fit_info", None) is not None:
                record["failed_fit"] = active_fit | error.fit_info
            save_progress(results, rows, class_rows, record, staging)
            raise
    print("Saved full-model validation tables, selected parameters and 24 selected pipelines.")
    print("No test scoring or feature selection performed.")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), default="iec104")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace full-training outputs only.")
    args = parser.parse_args()
    try:
        train(overwrite=args.overwrite, protocol=args.protocol)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Full-model training stopped: {error}\n")


if __name__ == "__main__":
    main()

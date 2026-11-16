"""Small shared checks that stop processing when a required safeguard fails."""


def require(condition, message):
    """Reject invalid inputs or state without silently relaxing the method."""
    if not condition:
        raise ValueError(message)


def complete_record(path):
    """Read settings only when the producing operation completed successfully."""
    import json
    require(path.is_file(), f"Required settings are missing: {path}")
    value = json.loads(path.read_text())
    require(value.get("status") == "complete", f"Required outputs are not current and complete: {path}")
    return value


def invalidate_results(root, protocol, stage, selector=None):
    """Mark dependent completion records stale before replacing their inputs.

    Numeric outputs and fitted models are retained. Consumers refuse stale
    records, and the consolidated table/figure is removed until regenerated.
    """
    import json
    from pathlib import Path
    root = Path(root)
    base = root / f"results/{protocol}"
    require(stage in {"preparation", "partition", "preprocessing", "full", "selection", "reduced", "evaluation", "benchmark"},
            "Unknown output dependency stage.")
    paths = []
    if stage in {"preparation", "partition", "preprocessing"}:
        paths.append(base / "validation/selected_parameters.json")
    if stage in {"preparation", "partition", "preprocessing", "full"}:
        paths.append(base / "validation/prediction_recovery.json")
        for suffix in ("", "sensitivity/"):
            paths.extend(base / f"{suffix}{name}" for name in
                ("feature_selection/selection.json", "validation/reduced_training.json", "test/evaluation.json"))
    elif stage == "selection":
        suffix = "sensitivity/" if selector else ""
        paths.extend(base / f"{suffix}{name}" for name in ("validation/reduced_training.json", "test/evaluation.json"))
        if selector is None:
            # Matched MI depends on the main selection's saved foreground rows.
            paths.extend(base / f"sensitivity/{name}" for name in
                ("feature_selection/selection.json", "validation/reduced_training.json", "test/evaluation.json"))
    elif stage == "reduced":
        paths.append(base / ("sensitivity/test/evaluation.json" if selector else "test/evaluation.json"))
    if stage != "benchmark":
        paths.append(base / "sensitivity/benchmark/benchmark.json")
    if stage in {"preparation", "partition", "preprocessing", "full", "selection"}:
        paths.append(base / "sensitivity/ranking_consistency.json")
    paths.append(base / "sensitivity/analysis.json")
    for path in paths:
        if path.exists():
            value = json.loads(path.read_text())
            value.update(status="stale", stale_reason=f"Rebuild required after {stage} outputs are replaced.")
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
    for path in (root / "results/comparison/reduction_summary.csv",
                 root / "figures/comparison/performance_prediction_tradeoffs.png"):
        path.unlink(missing_ok=True)


def selected_field_map(root, protocol, selector=None):
    """Resolve the actual ranked fields, rather than relying on settings alone."""
    import math
    from pathlib import Path
    import pandas as pd
    from src import settings
    suffix = "sensitivity/" if selector else ""
    table = pd.read_csv(Path(root) / f"results/{protocol}/{suffix}feature_selection/selected_features.csv")
    result = {}
    for (scope, method, fraction), part in table.groupby(["scope", "selector", "retention_fraction"], sort=True):
        part = part.sort_values("selection_order")
        count = math.ceil(float(fraction) * int(part.active_original_fields.iloc[0]))
        require(len(part) == count and part.field.is_unique
                and part.selection_order.tolist() == list(range(1, count+1)), "Selected-field cutoff or order differs.")
        result[f"{scope}/{method}_{int(100*fraction)}"] = part.field.tolist()
    methods = (selector,) if selector else ("shap", "mi")
    scopes = [*(f"fold_{i}" for i in range(settings.VALIDATION_FOLDS)), "development"]
    expected = {f"{scope}/{method}_{int(100*q)}" for scope in scopes
                for method in methods for q in settings.RETENTION_FRACTIONS}
    require(set(result) == expected, "Selected-field scope coverage differs.")
    return result


def validate_fitted_parameters(pipeline, expected, fit):
    """Allow only recorded convergence retries, preserving the selected recipe."""
    import json
    from src import settings
    expected = dict(expected)
    require(json.loads(fit["parameters"]) == expected, "Fitting record parameters differ from the selected recipe.")
    if pipeline.model_name == "logistic_regression":
        cap = int(fit["effective_max_iter"])
        require(cap in settings.LR_ITERATION_CAPS
                and int(fit["fit_attempts"]) == settings.LR_ITERATION_CAPS.index(cap)+1,
                "LR iteration ceiling is not a permitted recorded retry.")
        expected["max_iter"] = cap
    require(pipeline.parameters == expected, "Saved effective parameters differ from the fitting record.")
    actual = pipeline.estimator.get_params(deep=False)
    require(all(actual.get(key) == value for key, value in expected.items()),
            "Estimator parameters differ from the saved pipeline.")
    require(actual.get("random_state") == settings.ESTIMATOR_SEED,
            "Estimator seed differs from the selected fitting recipe.")


def same_fitted_state(left, right):
    """Compare learned mappings and arrays, including separately loaded encoders."""
    import numpy as np
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_fitted_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(same_fitted_state(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and all(same_fitted_state(a, b) for a, b in zip(left.flat, right.flat))
    if hasattr(left, "get_params") and hasattr(right, "get_params"):
        return type(left) is type(right) and same_fitted_state(vars(left), vars(right))
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        if np.isnan(left) and np.isnan(right):
            return True
    return bool(left == right)


def validate_reduction(root, protocol, selector=None):
    """Check selection, fitted inputs and parameters on the actual saved models."""
    from pathlib import Path
    import joblib
    import pandas as pd
    from src import settings
    from src.evaluate_validation import full_record
    from src.train_reduced_models import reduction_recipe, selection_recipe, subset_processor
    root = Path(root)
    suffix = "sensitivity/" if selector else ""
    folder = root / f"results/{protocol}/{suffix}"
    selection = complete_record(folder / "feature_selection/selection.json")
    require(selection["recipe"] == selection_recipe(protocol, selector), "Selection recipe is out of date.")
    full = full_record(root, protocol)
    record = complete_record(folder / "validation/reduced_training.json")
    require(record["recipe"] == reduction_recipe(protocol, selector), "Reduced training recipe is out of date.")
    if "full_training_recipe" in record:
        require(record["full_training_recipe"] == full["recipe"], "Full training recipe differs.")
    parameters = {m: full["selected"][m]["parameters"] for m in settings.CLASSIFIER_PANEL}
    if "full_selected_parameters" in record:
        require(record["full_selected_parameters"] == parameters, "Selected full parameters differ.")
    fields = selected_field_map(root, protocol, selector)
    if "selected_fields" in record:
        require(record["selected_fields"] == fields, "Selected fields changed after reduced fitting.")
    fits = pd.read_csv(folder / "validation/reduced_models.csv").set_index(["model", "condition", "scope"])
    require(fits.index.is_unique, "Fitting rows are repeated.")
    expected_keys = {(m, condition, scope) for entry in fields for scope, condition in [entry.split("/", 1)]
                     for m in settings.CLASSIFIER_PANEL}
    actual_keys = {(e["model"], e["condition"], e["scope"]) for e in record["pipelines"]}
    require(actual_keys == expected_keys and len(record["pipelines"]) == len(expected_keys)
            and set(fits.index) == expected_keys, "Fitted pipeline coverage differs.")
    full_processors = {}
    for entry in record["pipelines"]:
        model, condition, scope = entry["model"], entry["condition"], entry["scope"]
        path = root / entry["path"]
        require(path == root / f"models/{protocol}/reduced/{condition}/{model}/{scope}.joblib"
                and path.is_file(), "Saved reduced pipeline path differs.")
        pipeline = joblib.load(path)
        require(pipeline.model_name == model and pipeline.training_scope == scope
                and pipeline.classes_ == tuple(settings.protocol_settings(protocol).CLASSES)
                and set(pipeline.preprocessor.fields) == set(fields[f"{scope}/{condition}"]),
                "Saved reduced input fields or scope differ from current selection.")
        if (model, scope) not in full_processors:
            full_pipeline = joblib.load(root / full["selected"][model]["pipelines"][scope])
            full_processors[model, scope] = full_pipeline.preprocessor
        expected_processor = subset_processor(full_processors[model, scope], fields[f"{scope}/{condition}"])
        require(pipeline.preprocessor.fields == expected_processor.fields
                and same_fitted_state(pipeline.preprocessor.states_, expected_processor.states_)
                and pipeline.preprocessor.feature_map_ == expected_processor.feature_map_,
                "Reduced preprocessing differs from the fitted full state.")
        validate_fitted_parameters(pipeline, parameters[model], fits.loc[model, condition, scope])
    # Return verified scientific inputs even when they were not stored explicitly.
    record = record | {"full_training_recipe": full["recipe"], "full_selected_parameters": parameters,
                       "selected_fields": fields}
    return record, fields


def validate_evaluation(root, protocol, selector=None):
    """Reject results whose current selection, model recipe or population differs."""
    from pathlib import Path
    from src.evaluate_test import plan
    from src.evaluate_validation import full_record
    root = Path(root)
    suffix = "sensitivity/" if selector else ""
    reduction, fields = validate_reduction(root, protocol, selector)
    record = complete_record(root / f"results/{protocol}/{suffix}test/evaluation.json")
    require(record["plan"] == plan(protocol, selector), "Evaluation settings are out of date.")
    if "selected_fields" in record:
        require(record["selected_fields"] == fields, "Evaluation used a different selected input set.")
    if "reduced_training_recipe" in record:
        require(record["reduced_training_recipe"] == reduction["recipe"], "Evaluation fitting recipe differs.")
    if "full_training_recipe" in record:
        require(record["full_training_recipe"] == full_record(root, protocol)["recipe"], "Evaluation full-model recipe differs.")
    expected = {(e["model"], e["condition"]) for e in reduction["pipelines"] if e["scope"] == "development"}
    expected |= {(m, "full") for m in (*plan(protocol, selector)["models"], "dummy")}
    actual = {(e["model"], e["condition"]) for e in record["completed_conditions"]}
    require(expected == actual and len(actual) == len(record["completed_conditions"]), "Evaluation pipeline coverage differs.")
    return record, {"full_training_recipe": full_record(root, protocol)["recipe"],
                    "reduced_training_recipe": reduction["recipe"], "selected_fields": fields}


def validate_benchmark(root, protocol, record=None):
    from pathlib import Path
    root = Path(root)
    record = record or complete_record(root / f"results/{protocol}/sensitivity/benchmark/benchmark.json")
    require(record["status"] == "complete", "Prediction benchmark is not current and complete.")
    inputs = {}
    for selector, suffix in ((None, ""), ("mi_matched", "sensitivity/")):
        if (root / f"results/{protocol}/{suffix}test/evaluation.json").exists():
            _, inputs[selector or "main"] = validate_evaluation(root, protocol, selector)
    require(record.get("evaluation_inputs") == inputs,
            "Benchmark input selections or evaluation recipes changed; repeat the benchmark.")
    return record

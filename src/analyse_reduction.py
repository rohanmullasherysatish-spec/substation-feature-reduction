"""Explain reduced-input errors and account for selection/refitting overhead.

All diagnostics use the fixed observations, fitted transformations and saved
predictions. They never change allocations, fit classifiers or choose a winner.
"""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src import settings
from src.checks import require, validate_evaluation, validate_benchmark, selected_field_map
from src.train_full_models import atomic_json, load_training_inputs
from src.train_reduced_models import subset_processor


def signature_groups(processor, training, evaluation, labels, targets, standardize, class_count, protocol):
    """Describe float64 encoded equality, without using labels to form groups."""
    x = processor.transform(training[list(processor.fields)], standardize=standardize)
    z = processor.transform(evaluation[list(processor.fields)], standardize=standardize)
    if protocol == "iec61850":
        # Reuse the packet overlap routine's complete-row byte comparison.
        # This avoids a pandas grouping key for each of hundreds of columns.
        matrix = np.ascontiguousarray(np.concatenate([x, z]))
        require(matrix.dtype == np.float64 and np.isfinite(matrix).all(),
                "Packet matching requires finite float64 encoded inputs.")
        matrix[matrix == 0] = 0.0  # Signed zeros represent the same numeric value.
        vectors = matrix.view(np.dtype((np.void, matrix.shape[1]*matrix.dtype.itemsize))).ravel()
        _, groups = np.unique(vectors, return_inverse=True)
        del matrix, vectors
    else:
        combined = pd.DataFrame(np.concatenate([x, z]))
        groups = combined.groupby(list(combined.columns), sort=False, dropna=False).ngroup().to_numpy(dtype=int)
    train_groups, evaluation_groups = groups[:len(x)], groups[len(x):]
    counts = np.zeros((int(groups.max())+1, class_count), dtype=np.int32)
    np.add.at(counts, (train_groups, labels), 1)
    present = counts.sum(axis=1) > 0
    mixed = (counts > 0).sum(axis=1) > 1
    matched = present[evaluation_groups]
    return {"matched": matched, "mixed": mixed[evaluation_groups] & matched,
            "target_absent": matched & (counts[evaluation_groups, targets] == 0),
            "training_signature_groups": int(present.sum()),
            "mixed_label_training_signature_groups": int(mixed.sum())}


def collision_errors(root, protocol, destination):
    """Stratify saved errors by representation matching and supplied class label."""
    inputs, y, scopes, processors = load_training_inputs(root, protocol)
    assignments = pd.read_parquet(root / f"data/{protocol}/assignments.parquet")
    development = assignments.loc[assignments.partition.eq("development")].reset_index(drop=True)
    if protocol == "iec104":
        from src.iec104.partition import read_observations
        observations = read_observations(root)
    else:
        from src.iec61850.partition import read_observations
        observations, _ = read_observations(root)
    test = observations.loc[assignments.partition.eq("test")].reset_index(drop=True)
    test_metadata = assignments.loc[assignments.partition.eq("test")].reset_index(drop=True)
    classes = settings.protocol_settings(protocol).CLASSES
    field_maps = {"main": selected_field_map(root, protocol),
                  "mi_matched": selected_field_map(root, protocol, "mi_matched")}
    conditions = ["full", *(f"{m}_{int(100*q)}" for m in ("shap", "mi", "mi_matched")
                            for q in settings.RETENTION_FRACTIONS)]
    rows, class_rows, source_rows = [], [], []

    def error_row(signature, wrong, mask):
        matched = signature["matched"] & mask
        unmatched = ~signature["matched"] & mask
        n, nm, nu = int(mask.sum()), int(matched.sum()), int(unmatched.sum())
        errors = int((wrong & mask).sum())
        em, eu = int((wrong & matched).sum()), int((wrong & unmatched).sum())
        require(em+eu == errors and nm+nu == n, "Matching-stratum counts do not sum to the population.")
        return {"evaluation_rows": n, "matched_rows": nm, "unmatched_rows": nu,
                "matched_mixed_label_rows": int((signature["mixed"] & mask).sum()),
                "matched_target_absent_rows": int((signature["target_absent"] & mask).sum()),
                "errors": errors, "matched_errors": em, "unmatched_errors": eu,
                "error_rate": errors/n if n else np.nan,
                "matched_error_rate": em/nm if nm else np.nan,
                "unmatched_error_rate": eu/nu if nu else np.nan}

    for scope, training_mask in scopes.items():
        training = inputs.loc[training_mask]
        if scope == "development":
            population = "test"
            frame, metadata = test, test_metadata
            targets = pd.Categorical(metadata.class_label, categories=classes).codes.astype(int)
        else:
            population = "validation"
            frame = inputs.loc[~training_mask].reset_index(drop=True)
            metadata = development.loc[~training_mask].reset_index(drop=True)
            targets = y[~training_mask]
        for condition in conditions:
            additional = condition.startswith("mi_matched_")
            suffix = "sensitivity/" if additional else ""
            if condition == "full":
                processor = processors[scope]
            else:
                source = "mi_matched" if additional else "main"
                processor = subset_processor(processors[scope], field_maps[source][f"{scope}/{condition}"])
            prediction_name = ("test_predictions.parquet" if population == "test" else
                               "full_validation_predictions.parquet" if condition == "full" else
                               "reduced_validation_predictions.parquet")
            predictions = pd.read_parquet(root / f"data/{protocol}/{suffix}{prediction_name}",
                filters=[("scope", "=", "test" if population == "test" else scope), ("condition", "=", condition)],
                columns=["model", "record_id", "true_class", "predicted_class"])
            signatures = {view: signature_groups(processor, training, frame, y[training_mask], targets,
                           view == "standardised", len(classes), protocol) for view in ("unscaled", "standardised")}
            require(set(predictions.model) >= set(settings.CLASSIFIER_PANEL), "Saved diagnostic model coverage differs.")
            for model in settings.CLASSIFIER_PANEL:
                part = predictions.loc[predictions.model.eq(model)]
                require(np.array_equal(part.record_id, metadata.record_id)
                        and np.array_equal(part.true_class, metadata.class_label), "Diagnostic prediction identities differ.")
                view = "standardised" if model in settings.STANDARDIZE_NUMERIC_FOR else "unscaled"
                signature = signatures[view]
                wrong = part.predicted_class.to_numpy() != part.true_class.to_numpy()
                keys = {"population": population, "scope": scope, "model": model, "condition": condition,
                        "numeric_view": view, "comparison_precision": "float64",
                        "active_fields": len(processor.active_fields_), "encoded_columns": processor.n_features_out_}
                rows.append(keys | {k: signature[k] for k in
                    ("training_signature_groups", "mixed_label_training_signature_groups")}
                    | error_row(signature, wrong, np.ones(len(targets), dtype=bool)))
                for code, label in enumerate(classes):
                    class_rows.append(keys | {"class_label": label} | error_row(signature, wrong, targets == code))
                if protocol == "iec61850":
                    # The pooled normal label combines distinct legitimate scenarios.
                    for (source_file, label), indices in metadata.groupby(["source_file", "class_label"], sort=True).indices.items():
                        mask = np.zeros(len(metadata), dtype=bool)
                        mask[indices] = True
                        source_rows.append(keys | {"source_file": source_file, "class_label": label,
                            "evaluation_groups": int(metadata.loc[mask, "group_id"].nunique())}
                            | error_row(signature, wrong, mask))
            print(f"{scope}, {condition}: actual saved errors stratified by encoded matching.", flush=True)
    pd.DataFrame(rows).to_csv(destination / "representation_errors.csv", index=False)
    pd.DataFrame(class_rows).to_csv(destination / "representation_class_errors.csv", index=False)
    if source_rows:
        pd.DataFrame(source_rows).to_csv(destination / "representation_source_errors.csv", index=False)
    require(len(rows) == len(scopes)*len(conditions)*len(settings.CLASSIFIER_PANEL)
            and len(class_rows) == len(rows)*len(classes), "Representation diagnostic coverage differs.")
    return len(rows), len(class_rows), len(source_rows)


def cost_accounting(root, protocol, destination):
    """Calculate conditional, measured-cost offsets; do not estimate deployment ROI."""
    base = root / f"results/{protocol}"
    primary = pd.read_csv(base / "test/point_metrics.csv")
    extra = pd.read_csv(base / "sensitivity/test/point_metrics.csv")
    points = pd.concat([primary, extra.loc[extra.condition.str.startswith("mi_matched_")]], ignore_index=True)
    points = points.set_index(["model", "condition"])
    benchmark = base / "sensitivity/benchmark"
    times = pd.read_csv(benchmark / "prediction_summary.csv").set_index(["model", "condition"])
    raw = pd.read_csv(benchmark / "prediction_times.csv").pivot(index="round", columns=["model", "condition"], values="seconds")
    records = int(times.records.iloc[0])
    source = {"main": json.loads((base / "feature_selection/selection.json").read_text()),
              "mi_matched": json.loads((base / "sensitivity/feature_selection/selection.json").read_text())}
    models = list(settings.CLASSIFIER_PANEL)
    rows = []
    for selector in ("shap", "mi", "mi_matched"):
        analyses = source["mi_matched" if selector == "mi_matched" else "main"]["analyses"]
        analyses = [a for a in analyses if (a.get("model") in models if selector == "shap" else a.get("method") == selector)]
        def seconds(a):
            return a.get("seconds_including_transform_and_save", a.get("seconds_including_transform", 0.))
        selection = sum(seconds(a) for a in analyses if a["scope"] == "development")
        all_scopes = sum(seconds(a) for a in analyses)
        for fraction in settings.RETENTION_FRACTIONS:
            condition = f"{selector}_{int(100*fraction)}"
            for chosen in ([m] for m in models):
                rows.append(cost_row(points, times, raw, chosen, condition, records, selection, all_scopes))
            rows.append(cost_row(points, times, raw, models, condition, records, selection, all_scopes))
    table = pd.DataFrame(rows)
    require(len(table) == 36, "Cost-accounting scope coverage differs.")
    table.to_csv(destination / "cost_accounting.csv", index=False)
    return len(table)


def cost_row(points, times, raw, models, condition, records, selector_seconds, all_scope_seconds):
    before = sum(times.loc[m, "full"]["median_seconds"] for m in models)
    after = sum(times.loc[m, condition]["median_seconds"] for m in models)
    saving = before - after
    reduced_fit = sum(points.loc[m, condition]["development_fit_seconds"] for m in models)
    full_fit = sum(points.loc[m, "full"]["development_fit_seconds"] for m in models)
    paired_savings = sum(raw[m, "full"] - raw[m, condition] for m in models)
    row = {"usage_scope": "single_classifier" if len(models) == 1 else "five_classifier_panel",
           "model": models[0] if len(models) == 1 else "all_five", "model_count": len(models),
           "condition": condition, "task_observations": records,
           "full_prediction_seconds": before, "reduced_prediction_seconds": after,
           "prediction_seconds_saved_per_task": saving,
           "paired_saved_seconds_q25": paired_savings.quantile(.25),
           "paired_saved_seconds_median": paired_savings.median(),
           "paired_saved_seconds_q75": paired_savings.quantile(.75),
           "shared_selector_seconds_development": selector_seconds,
           "shared_selector_seconds_all_training_scopes": all_scope_seconds,
           "full_development_fit_seconds": full_fit, "reduced_development_fit_seconds": reduced_fit,
           "incremental_selection_and_refit_seconds": selector_seconds+reduced_fit,
           "minimum_macro_f1_change": min(points.loc[m, condition]["macro_f1"]-points.loc[m, "full"]["macro_f1"] for m in models),
           "offset_status": "positive_observed_prediction_saving" if saving > 0 else "no_observed_prediction_saving"}
    for name, overhead in (("selector_only", selector_seconds), ("selection_and_refit", selector_seconds+reduced_fit)):
        tasks = math.ceil(overhead/saving) if saving > 0 else None
        row[name+"_task_repetitions_to_offset"] = tasks
        row[name+"_observation_equivalents"] = tasks*records if tasks is not None else None
        row[name+"_classifier_prediction_equivalents"] = tasks*records*len(models) if tasks is not None else None
    return row


def analyse(root=settings.PROJECT_ROOT, protocol="iec104", overwrite=False):
    root = Path(root)
    extra = root / f"results/{protocol}/sensitivity"
    destination = extra / "diagnostics"
    names = ("representation_errors.csv", "representation_class_errors.csv", "cost_accounting.csv")
    if protocol == "iec61850":
        names += ("representation_source_errors.csv",)
    require(overwrite or not any((destination / name).exists() for name in names),
            "Reduction diagnostics exist; use --overwrite deliberately.")
    evaluations = {key: validate_evaluation(root, protocol, selector)[1]
                   for key, selector in (("main", None), ("mi_matched", "mi_matched"))}
    benchmark = validate_benchmark(root, protocol)
    destination.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "protocol": protocol, "evaluation_inputs": evaluations,
        "comparison_precision": "float64 encoded diagnostic in each model's preprocessing view; native float32 diagnostics remain in evaluation overlap tables",
        "collision_interpretation": "descriptive strata with different class compositions; matching alone does not demonstrate leakage or cause every error",
        "cost_assumptions": "full classifier panel and selected settings already available; repeated identical warm batch, hardware and distribution; selector charged once within each row, never summed across deployment alternatives",
        "cost_boundaries": "selection and one reduced development refit; no acquisition, tuning, full-panel creation, retraining schedule, energy or operating costs; fitting times are individual recorded fits",
        "cost_interpretation": "illustrative measured-cost offsets, not operational break-even or evidence of performance equivalence",
        "benchmark_repeats": benchmark["repeats"], "models_fitted": 0, "predictions_generated": 0}
    with TemporaryDirectory(prefix=f".validation_{protocol}_", dir=root) as folder:
        staging = Path(folder)
        atomic_json(extra / "analysis.json", record, staging)
        try:
            overall, classes, sources = collision_errors(root, protocol, staging)
            costs = cost_accounting(root, protocol, staging)
            for name in names:
                (staging / name).replace(destination / name)
            record.update(status="complete", representation_rows=overall, representation_class_rows=classes,
                          representation_source_rows=sources, cost_rows=costs)
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(extra / "analysis.json", record, staging)
    print(f"Saved {overall} representation summaries, {classes} class rows, {sources} source rows and {costs} cost scenarios; no fitting or predictions.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    analyse(protocol=args.protocol, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

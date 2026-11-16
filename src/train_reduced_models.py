"""Fit declared SHAP/MI reductions using fixed recipes and saved development folds.

Full-input fits, allocation and learned field transformations are reused.
All test rows remain outside fitting, predictions and comparisons.
"""

import argparse
import importlib.metadata
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import settings
from src.evaluate_validation import confusion_rows, full_record, prediction_table, timed_probabilities
from src.modelling import classification_measures
from src.checks import require, invalidate_results, selected_field_map
from src.train_full_models import atomic_json, fit_and_measure, load_training_inputs, save_pipeline, protocol_classes


def selection_recipe(protocol, selector=None):
    """Keep each protocol's declared MI representation and selection settings."""
    if selector == "mi_matched":
        from src.selection_checks import matched_recipe
        return matched_recipe(protocol)
    if protocol == "iec104":
        from src.iec104.feature_selection import selection_recipe as recipe
    else:
        from src.iec61850.feature_selection import selection_recipe as recipe
    protocol_classes(protocol)
    return recipe()


def overlap_count(training, validation, dtype, protocol):
    if protocol == "iec61850":
        from src.iec61850.preprocessing import exact_overlap
        return exact_overlap(training, validation, dtype)
    from src.iec104.preprocessing import matching_rows
    return matching_rows(training.astype(dtype, copy=False), validation.astype(dtype, copy=False))


def subset_processor(processor, selected):
    """Project a fitted training state; never learn new statistics from other rows.

    Preserve original column order so projection exactly matches the full
    matrix's selected components.
    """
    selected = set(selected)
    require(bool(selected) and selected.issubset(processor.active_fields_), "Invalid reduced field subset.")
    result = deepcopy(processor)
    result.states_ = [s for s in result.states_ if s["field"] in selected]
    result.fields = result.active_fields_ = tuple(s["field"] for s in result.states_)
    result.field_records_ = [r for r in result.field_records_ if r["original_field"] in selected]
    result.category_levels_ = {f: levels for f, levels in result.category_levels_.items() if f in selected}
    result.feature_map_ = [dict(entry) for entry in result.feature_map_ if entry["original_field"] in selected]
    for index, entry in enumerate(result.feature_map_):
        entry["column_index"] = index
    result.n_features_out_ = len(result.feature_map_)
    # Packet states also retain typed rules and pooled vocabularies. Project
    # those interfaces together; no statistics are refitted for the subset.
    if hasattr(result, "rules"):
        result.rules = [r for r in result.rules if r["field"] in selected]
    if hasattr(result, "vocabularies_"):
        result.vocabularies_ = {f: value for f, value in result.vocabularies_.items() if f in selected}
    for attribute in ("component_screen_", "dropped_fields_"):
        if hasattr(result, attribute):
            setattr(result, attribute, [r for r in getattr(result, attribute) if r["field"] in selected])
    return result


def reduction_recipe(protocol="iec104", selector=None):
    return {"selection_recipe": selection_recipe(protocol, selector), "parameter_policy": settings.REDUCED_PARAMETER_POLICY,
            "seed": settings.ESTIMATOR_SEED, "prediction_repeats": settings.PREDICTION_TIMING_REPEATS,
            "comparison": "same saved observations and allocation; reduced minus full; no equivalence pass/fail",
            "test_scoring": False}


def train(root=settings.PROJECT_ROOT, overwrite=False, protocol="iec104", selector=None):
    root = Path(root).resolve()
    classes = protocol_classes(protocol)
    require(selector in (None, "mi_matched"), "Unknown additional selector.")
    selectors = (selector,) if selector else ("shap", "mi")
    suffix = "sensitivity/" if selector else ""
    result_root = root / f"results/{protocol}/{suffix}"
    reference_results = root / f"results/{protocol}/validation"
    results = result_root / "validation"
    results.mkdir(parents=True, exist_ok=True)
    filenames = ("reduced_models.csv", "reduced_class_metrics.csv", "reduced_selected_metrics.csv",
                 "reduced_confusion_matrices.csv", "reduction_comparisons.csv", "class_reduction_comparisons.csv",
                 "representation_overlap.csv", "source_reduction_comparisons.csv", "reduced_training.json")
    predictions_path = root / f"data/{protocol}/{suffix}reduced_validation_predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [root / f"models/{protocol}/reduced/{selector}_{int(100 * fraction)}/{name}/{scope}.joblib"
               for selector in selectors for fraction in settings.RETENTION_FRACTIONS
               for name in settings.CLASSIFIER_PANEL
               for scope in [*(f"fold_{i}" for i in range(settings.VALIDATION_FOLDS)), "development"]]
    protected = [*(results / f for f in filenames), predictions_path, *targets]
    require(all(p.resolve() == p and (not p.exists() or p.is_file()) for p in protected),
            "A reduced output is redirected or is not a regular file.")
    require(overwrite or not any(p.exists() for p in protected),
            "Reduced outputs exist; use --overwrite deliberately.")
    record = full_record(root, protocol)
    selection = json.loads((result_root / "feature_selection/selection.json").read_text())
    require(selection["status"] == "complete" and selection["recipe"] == selection_recipe(protocol, selector), "Selection is incomplete or settings changed.")
    require(json.loads((reference_results / "prediction_recovery.json").read_text())["status"] == "complete", "Recover full predictions first.")
    require(settings.REDUCED_PARAMETER_POLICY == "reuse_selected_full_parameters", "Unexpected reduced tuning policy.")
    if protocol == "iec61850" or selector:
        require(selection.get("full_training_recipe") == record["recipe"], "Selection used a different full representation or recipe.")
        expected_parameters = {name: record["selected"][name]["parameters"] for name in settings.CLASSIFIER_PANEL}
        require(selection.get("selected_full_parameters") == expected_parameters,
                "Full-model parameters changed after feature selection.")
    for name, version in selection["packages"].items():
        require(importlib.metadata.version(name) == version, f"Selection package version changed: {name}")
    inputs, labels, scopes, processors = load_training_inputs(root, protocol)
    require(set(selection["completed_scopes"]) == set(scopes), "Selection scope coverage differs.")
    metadata = pd.read_parquet(root / f"data/{protocol}/assignments.parquet")
    metadata = metadata.loc[metadata.partition.eq("development")].reset_index(drop=True)
    sampled = pd.read_parquet(root / f"data/{protocol}/{suffix}feature_selection/samples.parquet")
    identities = ["record_id", "dataset_row", "source_file", "source_row", "group_id", "class_label"]
    for scope, mask in scopes.items():
        for role in (("mi_matched",) if selector else ("foreground", "background", "mi")):
            block = sampled[(sampled.scope == scope) & (sampled.role == role)]
            positions = block.development_position.to_numpy(dtype=int)
            require(len(positions) > 0 and mask[positions].all()
                    and metadata.iloc[positions][identities].reset_index(drop=True).equals(block[identities].reset_index(drop=True)),
                    "Selection sample identities or training boundaries differ.")
    features = pd.read_csv(result_root / "feature_selection/selected_features.csv")
    common_rank = None if selector else pd.read_csv(result_root / "feature_selection/common_ranking.csv")
    independent_rank = pd.read_csv(result_root / "feature_selection/rankings.csv")
    full_metrics = pd.read_csv(reference_results / "full_selected_metrics.csv").set_index(["model", "scope"])
    full_classes = pd.read_csv(reference_results / "full_selected_class_metrics.csv").set_index(["model", "scope", "class_label"])
    fit_rows, class_rows, metrics, confusions, comparisons, class_comparisons = [], [], [], [], [], []
    overlaps, projected = {}, {}
    source_comparisons = []
    status = {"status": "running", "protocol": protocol, "recipe": reduction_recipe(protocol, selector),
              "selected_fields": selected_field_map(root, protocol, selector),
              "full_training_recipe": record["recipe"],
              "full_selected_parameters": {name: record["selected"][name]["parameters"]
                                           for name in settings.CLASSIFIER_PANEL},
              "pipelines": [], "logical_fits_completed": 0, "test_scoring": False}

    def compare(name, scope, condition, fraction, overall, per_class, values, population_mask, reference_predictions):
        full = full_metrics.loc[name, scope]
        expected = reference_predictions.loc[metadata.loc[population_mask, "record_id"]]
        require(np.array_equal(expected.true_class.to_numpy(), metadata.loc[population_mask, "class_label"].to_numpy()),
                "Paired full/reduced labels differ.")
        old_correct = expected.predicted_class.to_numpy() == expected.true_class.to_numpy()
        new_correct = values.argmax(axis=1) == labels[population_mask]
        require(int(np.sum(old_correct & ~new_correct) - np.sum(~old_correct & new_correct))
                == overall["errors"] - int(full.errors), "Paired error changes do not match metric differences.")
        row = {"model": name, "condition": condition, "scope": scope, "retention_fraction": fraction,
               "new_errors": int(np.sum(old_correct & ~new_correct)), "corrected_errors": int(np.sum(~old_correct & new_correct)),
               "both_wrong": int(np.sum(~old_correct & ~new_correct))}
        for key in ("accuracy", "macro_f1", "weighted_f1", "normal_false_alarm_rate", "errors"):
            row.update({"full_" + key: full[key], "reduced_" + key: overall[key], "difference_" + key: overall[key] - full[key]})
        comparisons.append(row)
        # Source identities are analysis metadata only, never classifier inputs.
        population = metadata.loc[population_mask].copy()
        population["full_wrong"] = ~old_correct
        population["reduced_wrong"] = ~new_correct
        population["new_wrong"] = old_correct & ~new_correct
        population["corrected"] = ~old_correct & new_correct
        population["both_wrong"] = ~old_correct & ~new_correct
        for (source, label), frame in population.groupby(["source_file", "class_label"], sort=True):
            n = len(frame)
            before, after = int(frame.full_wrong.sum()), int(frame.reduced_wrong.sum())
            source_comparisons.append({"model": name, "condition": condition, "scope": scope,
                "source_file": source, "class_label": label, "records": n,
                "full_errors": before, "reduced_errors": after, "difference_errors": after-before,
                "new_errors": int(frame.new_wrong.sum()), "corrected_errors": int(frame.corrected.sum()),
                "both_wrong": int(frame.both_wrong.sum()),
                "full_error_rate": before/n, "reduced_error_rate": after/n,
                "difference_error_rate": (after-before)/n,
                "full_normal_false_alarm_rate": before/n if label == "normal" else None,
                "reduced_normal_false_alarm_rate": after/n if label == "normal" else None})
        for item in per_class:
            old = full_classes.loc[name, scope, item["class_label"]]
            require(item["support"] == old.support, "Full/reduced class support differs.")
            pair = {"model": name, "condition": condition, "scope": scope, "class_label": item["class_label"], "support": item["support"]}
            for key in ("precision", "recall", "f1", "errors"):
                pair.update({"full_" + key: old[key], "reduced_" + key: item[key], "difference_" + key: item[key] - old[key]})
            class_comparisons.append(pair)

    with TemporaryDirectory(prefix=f".reduce_{protocol}_", dir=root) as folder, ExitStack() as stack:
        staging = Path(folder)
        writer = None

        def progress():
            tables = (fit_rows, class_rows, metrics, confusions, comparisons, class_comparisons, list(overlaps.values()), source_comparisons)
            for filename, rows in zip(filenames[:-1], tables):
                pd.DataFrame(rows).to_csv(staging / filename, index=False)
                (staging / filename).replace(results / filename)
            status["logical_fits_completed"] = len(fit_rows)
            status["fit_attempts"] = sum(r["fit_attempts"] for r in fit_rows)
            status["fit_attempts"] += status.get("failed_fit", {}).get("fit_attempts", 0)
            atomic_json(results / "reduced_training.json", status, staging)

        invalidate_results(root, protocol, "reduced", selector)
        progress()
        try:
            for name in settings.CLASSIFIER_PANEL:
                reference_predictions = pq.read_table(root / f"data/{protocol}/full_validation_predictions.parquet",
                    columns=["record_id", "true_class", "predicted_class"], filters=[("model", "=", name)]).to_pandas().set_index("record_id")
                require(reference_predictions.index.is_unique and set(reference_predictions.index) == set(metadata.record_id),
                        "Full held-fold prediction coverage differs.")
                for selector in selectors:
                    for fraction in settings.RETENTION_FRACTIONS:
                        condition = f"{selector}_{int(100 * fraction)}"
                        pooled = np.empty((len(inputs), len(classes)), dtype=np.float64)
                        visited = np.zeros(len(inputs), dtype=bool)
                        for scope, mask in scopes.items():
                            full_processor = processors[scope]
                            chosen = features[(features.scope == scope) & (features.selector == selector)
                                              & np.isclose(features.retention_fraction, fraction)].sort_values("selection_order")
                            selected = chosen.field.tolist()
                            require(len(selected) == len(set(selected)) == math.ceil(fraction * len(full_processor.active_fields_)),
                                    "Selected input count differs from the declared budget.")
                            ranking = (common_rank[common_rank.scope == scope].sort_values("selection_order")
                                       if selector == "shap" else independent_rank[(independent_rank.scope == scope)
                                                                                  & (independent_rank.method == selector)].sort_values(["rank", "field"]))
                            require(set(ranking.field) == set(full_processor.active_fields_)
                                    and selected == ranking.field.tolist()[:len(selected)],
                                    "Selected fields do not match the saved ranked cutoff.")
                            key = (scope, condition)
                            probe_positions = np.concatenate([np.flatnonzero(mask & (labels == c))[:2]
                                                              for c in range(len(classes))])
                            probe = inputs.iloc[probe_positions]
                            if key not in projected:
                                processor = subset_processor(full_processor, selected)
                                fields = list(processor.fields)
                                for standardize in (False, True):
                                    np.testing.assert_array_equal(processor.transform(probe[fields], standardize=standardize),
                                        full_processor.transform(probe, standardize=standardize)[:, full_processor.field_columns(selected)])
                                processor.validate_training_frame(inputs.loc[mask, fields])
                                projected[key] = processor
                            processor = projected[key]
                            fields = list(processor.fields)
                            started = perf_counter()
                            xtrain = processor.transform_for_model(inputs.loc[mask, fields], name)
                            xvalid = processor.transform_for_model(inputs.loc[~mask, fields], name)
                            transform_seconds = perf_counter() - started
                            if scope != "development":
                                view = "standardised" if name in settings.STANDARDIZE_NUMERIC_FOR else "unscaled"
                                key = (scope, condition, view)
                                if key not in overlaps:
                                    overlaps[key] = {"scope": scope, "condition": condition, "numeric_view": view,
                                        "validation_rows": len(xvalid), "encoded_columns": xvalid.shape[1],
                                        "matching_training_float64": overlap_count(xtrain, xvalid, np.float64, protocol),
                                        "matching_training_float32": overlap_count(xtrain, xvalid, np.float32, protocol),
                                        "rows_removed": 0}
                            print(f"{name}, {condition}, {scope}: fitting {len(selected)} original inputs.", flush=True)
                            selected_record = record["selected"][name]
                            active_fit = {"model": name, "condition": condition, "scope": scope}
                            pipeline, row, per_class = fit_and_measure(name, selected_record["candidate_index"], scope,
                                selected_record["parameters"], processor, xtrain, labels[mask], xvalid, labels[~mask], transform_seconds, class_labels=classes)
                            row.update(condition=condition, selector=selector, retention_fraction=fraction)
                            fit_rows.append(row)
                            class_rows.extend(dict(r, condition=condition, selector=selector, retention_fraction=fraction) for r in per_class)
                            target = root / f"models/{protocol}/reduced/{condition}/{name}/{scope}.joblib"
                            save_pipeline(pipeline, target, staging, probe[fields])
                            full_path = root / record["selected"][name]["pipelines"][scope]
                            row.update(pipeline_bytes=target.stat().st_size,
                                       full_pipeline_bytes=full_path.stat().st_size,
                                       full_encoded_columns=full_processor.n_features_out_)
                            status["pipelines"].append({"model": name, "condition": condition, "scope": scope,
                                                        "path": target.relative_to(root).as_posix()})
                            if scope != "development":
                                evaluation = ~mask
                                values, timing = timed_probabilities(pipeline, inputs.loc[evaluation, fields])
                                overall, detail = classification_measures(labels[evaluation], values, class_labels=classes)
                                require(np.isclose(overall["macro_f1"], row["validation_macro_f1"], atol=1e-12, rtol=0),
                                        "Saved-pipeline and fit-time validation disagree.")
                                metrics.append({"model": name, "condition": condition, "scope": scope,
                                    "retention_fraction": fraction, "active_original_fields": len(selected),
                                    "encoded_columns": processor.n_features_out_, "model_bytes": target.stat().st_size,
                                    **overall, **timing})
                                arrow = pa.Table.from_pandas(prediction_table(metadata.loc[evaluation], values, name, scope, condition, class_labels=classes), preserve_index=False)
                                if writer is None:
                                    writer = stack.enter_context(pq.ParquetWriter(staging / predictions_path.name, arrow.schema))
                                writer.write_table(arrow)
                                require(not visited[evaluation].any(), "Reduced validation row evaluated twice.")
                                pooled[evaluation], visited[evaluation] = values, True
                                confusions.extend(confusion_rows(labels[evaluation], values, name, scope, condition, class_labels=classes))
                                compare(name, scope, condition, fraction, overall, detail, values, evaluation, reference_predictions)
                                print(f"  validation macro F1={overall['macro_f1']:.6f}; errors={overall['errors']}.", flush=True)
                            del pipeline, xtrain, xvalid
                            progress()
                        require(visited.all(), "Incomplete reduced held-fold coverage.")
                        overall, detail = classification_measures(labels, pooled, class_labels=classes)
                        metrics.append({"model": name, "condition": condition, "scope": "pooled_validation",
                                        "retention_fraction": fraction, **overall})
                        class_rows.extend({"model": name, "condition": condition, "scope": "pooled_validation",
                                           "population": "validation", **r} for r in detail)
                        confusions.extend(confusion_rows(labels, pooled, name, "pooled_validation", condition, class_labels=classes))
                        compare(name, "pooled_validation", condition, fraction, overall, detail, pooled,
                                np.ones(len(labels), dtype=bool), reference_predictions)
                        progress()
            conditions = len(settings.CLASSIFIER_PANEL) * len(selectors) * len(settings.RETENTION_FRACTIONS)
            expected_fits = conditions * len(scopes)
            expected_metrics = conditions * (settings.VALIDATION_FOLDS + 1)
            expected_classes = conditions * (2*settings.VALIDATION_FOLDS + 2) * len(classes)
            require(len(fit_rows) == expected_fits and len(class_rows) == expected_classes and len(metrics) == expected_metrics,
                    "Reduced fit/metric coverage differs from the declared comparison.")
            writer.close()
            (staging / predictions_path.name).replace(predictions_path)
            status["status"] = "complete"
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
            if getattr(error, "fit_info", None):
                status["failed_fit"] = active_fit | error.fit_info
            raise
        finally:
            progress()
    print(f"Completed {len(fit_rows)} reduced fits and saved pipelines, with paired validation comparisons; no test scoring.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), default="iec104")
    parser.add_argument("--selector", choices=("mi_matched",), help="Run only the additional matched-observation MI condition.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(overwrite=args.overwrite, protocol=args.protocol, selector=args.selector)


if __name__ == "__main__":
    main()

"""Select original packet fields using common SHAP ranks and an MI benchmark.

Reuse the saved full models and training-only preprocessing. Explanations and
MI estimates use each fold's training rows, then all development data. Model
recipes were selected using development validation; these are conditional
training analyses, not an independent assessment of the whole tuning process.
"""

import argparse
import importlib.metadata
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif
from threadpoolctl import threadpool_limits

from src import settings
from src.iec61850 import settings as cfg
from src.checks import require, invalidate_results
from src.feature_selection import (
    selection_recipe as shared_selection_recipe, stratified_rows, shap_ranking, consensus_table,
)
from src.evaluate_validation import full_record, saved_pipeline
from src.train_full_models import atomic_json, load_training_inputs


def selection_recipe():
    """Declare how structured packet fields become one selectable original unit."""
    recipe = shared_selection_recipe(
        "fitted logical components; nominal categories coded jointly after training-only pooling; "
        "numeric values use saved medians; count/byte/missingness components discrete")
    return recipe | {
        "classes": list(cfg.CLASSES),
        "mi_field_aggregation": "maximum component MI within each active original field; not joint multivariate MI",
        "mi_width_limitation": "maximum avoids sum/mean width scaling but wider fields have more opportunities for a high component score",
        "mi_sampling_interpretation": "same capped training populations as declared; not the class-balanced SHAP estimand",
        "expanded_field_policy": "retain every fitted encoded column of a selected original field; SV payload remains one field",
    }


def mi_inputs(frame, processor):
    """Use fitted logical components rather than fragmenting nominal categories.

    Numeric medians, category pooling and active component choices come from
    the saved training preprocessor. Integer coding changes only category names;
    those columns are marked discrete, so code magnitude implies no order.
    """
    processor._validate_frame(frame)
    rules = {r["field"]: r for r in processor.rules}
    arrays, records = [], []

    def append(field, component, encoding, values, discrete):
        values = np.asarray(values)
        levels, codes = np.unique(values, return_inverse=True)
        arrays.append(codes.astype(np.float64) if discrete else values.astype(np.float64))
        records.append({"field": field, "component": component, "encoding": encoding,
                        "discrete": bool(discrete), "sample_unique_values": len(levels),
                        "sample_rows": len(frame)})

    for field_state in processor.states_:
        field = field_state["field"]
        parts = processor._components(frame[field], rules[field])
        for state in field_state["components"]:
            name = state["component"]
            raw = parts[name]
            if state["kind"] == "categorical":
                values = processor._category_codes(raw, field_state["vocabulary"])
                append(field, name, "nominal_joint", values, True)
            else:
                absent = np.isnan(raw)
                if state["value_column"]:
                    values = np.where(absent, state["median"], raw)
                    append(field, name, "numeric", values, state["mi_discrete"])
                if state["missing_indicator"]:
                    append(field, name, "missing_indicator", absent.astype(np.int64), True)
    require(bool(arrays), "No fitted MI component is available.")
    matrix = np.column_stack(arrays)
    require(np.isfinite(matrix).all(), "Nonfinite MI components.")
    return matrix, pd.DataFrame(records)


def mi_ranking(frame, labels, processor):
    """Score a field by its strongest fitted component, without summing widths.

    This univariate benchmark does not estimate joint information in an entire
    payload/list. Maximum pooling can favour wider fields through extra chances;
    component counts and scores are saved so that limitation remains visible.
    """
    started = perf_counter()
    matrix, detail = mi_inputs(frame, processor)
    variable = detail.sample_unique_values.to_numpy() > 1
    scores = np.zeros(len(detail), dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=settings.TRAINING_WORKERS), joblib.parallel_backend("threading"):
        warnings.simplefilter("always")
        scores[variable] = mutual_info_classif(matrix[:, variable], labels,
            discrete_features=detail.discrete.to_numpy()[variable], n_neighbors=settings.MI_NEIGHBOURS,
            random_state=settings.MI_ESTIMATION_SEED, n_jobs=settings.TRAINING_WORKERS)
    require(np.isfinite(scores).all() and (scores >= 0).all(), "Invalid MI estimates.")
    detail["importance"] = scores
    rows = []
    for field in processor.active_fields_:
        group = detail.loc[detail.field.eq(field)]
        require(len(group) > 0, "An active field has no MI component.")
        best = group.sort_values(["importance", "component", "encoding"], ascending=[False, True, True]).iloc[0]
        rows.append({"field": field, "importance": best.importance,
                     "method": "mi", "model": "independent", "sample_rows": len(frame),
                     "scored_components": len(group), "sample_variable_components": int(group.sample_unique_values.gt(1).sum()),
                     "encoded_columns": len(processor.field_columns([field])),
                     "strongest_component": best.component, "strongest_encoding": best.encoding})
    table = pd.DataFrame(rows)
    table["rank"] = rankdata(-table.importance.to_numpy(), method="average")
    info = {"method": "mi", "seconds_including_transform": perf_counter() - started,
            "rows": len(frame), "logical_components": len(detail),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
    return table, detail, info


def select(root=settings.PROJECT_ROOT, overwrite=False):
    root = Path(root).resolve()
    results = root / "results/iec61850/feature_selection"
    names = ("rankings.csv", "common_ranking.csv", "importance_by_class.csv", "selected_features.csv", "mi_components.csv", "selection.json")
    local = root / "data/iec61850/feature_selection"
    expected_scopes = [*(f"fold_{i}" for i in range(settings.VALIDATION_FOLDS)), "development"]
    protected = [*(results / n for n in names), local / "samples.parquet"]
    protected += [local / "shap" / name / f"{scope}.npz"
                  for name in settings.CLASSIFIER_PANEL for scope in expected_scopes]
    require(all(p.resolve() == p and (not p.exists() or p.is_file()) for p in protected),
            "A selection output is redirected or is not a regular file.")
    require(overwrite or not any(p.exists() for p in protected), "Selection outputs exist; use --overwrite deliberately.")
    record = full_record(root, "iec61850")
    inputs, labels, scopes, processors = load_training_inputs(root, "iec61850")
    metadata = pd.read_parquet(root / "data/iec61850/assignments.parquet")
    metadata = metadata.loc[metadata.partition.eq("development")].reset_index(drop=True)
    samples, sample_positions = [], {}
    plans = [("foreground", settings.SHAP_FOREGROUND_PER_CLASS, settings.SELECTION_SAMPLE_SEED, 0),
             ("background", settings.SHAP_BACKGROUND_PER_CLASS, settings.SELECTION_SAMPLE_SEED, 1),
             ("mi", settings.MI_ROWS_PER_CLASS, settings.MI_SAMPLE_SEED, 2)]
    for index, (scope, mask) in enumerate(scopes.items()):
        for role, cap, seed, role_code in plans:
            selected = np.flatnonzero(mask)[stratified_rows(labels[mask], cap, seed, index, role_code, len(cfg.CLASSES))]
            require(mask[selected].all() and len(np.unique(selected)) == len(selected), "Invalid selection sample.")
            sample_positions[scope, role] = selected
            samples.append(metadata.iloc[selected].assign(scope=scope, role=role, development_position=selected))
    sample_table = pd.concat(samples, ignore_index=True)
    local.mkdir(parents=True, exist_ok=True)
    sample_table.to_parquet(local / "samples.parquet", index=False)
    results.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "recipe": selection_recipe(), "completed_scopes": [], "analyses": [],
              "full_training_recipe": record["recipe"],
              "selected_full_parameters": {name: values["parameters"] for name, values in record["selected"].items()
                                           if name in settings.CLASSIFIER_PANEL},
              "test_records_used": 0, "classifiers_fitted": 0,
              "packages": {n: importlib.metadata.version(n) for n in ("shap", "numpy", "scipy", "scikit-learn")}}
    ranks, common, class_rows, subsets, components = [], [], [], [], []
    with TemporaryDirectory(prefix=".selection_iec61850_", dir=root) as folder:
        staging = Path(folder)
        invalidate_results(root, 'iec61850', 'selection')
        atomic_json(results / "selection.json", status, staging)
        try:
            for scope, mask in scopes.items():
                processor = processors[scope]
                foreground, background, mi = [sample_positions[scope, role] for role in ("foreground", "background", "mi")]
                model_rankings = {}
                for name in settings.CLASSIFIER_PANEL:
                    print(f"{scope}, {name}: explaining {len(foreground)} training rows against {len(background)} background rows.", flush=True)
                    pipeline, _ = saved_pipeline(root, record, name, scope, processor, "iec61850")
                    table, per_class, info = shap_ranking(pipeline, inputs.iloc[foreground], inputs.iloc[background],
                        labels[foreground], processor, local / "shap" / name / f"{scope}.npz", cfg.CLASSES)
                    ranks.append(table.assign(scope=scope))
                    class_rows.extend(dict(row, scope=scope) for row in per_class)
                    model_rankings[name] = table
                    status["analyses"].append(dict(info, scope=scope))
                    atomic_json(results / "selection.json", status, staging)
                    print(f"  reconstruction error <= {info['maximum_additivity_absolute_error']:.3g}.", flush=True)
                combined = consensus_table(model_rankings)
                common.append(combined.assign(scope=scope))
                table, detail, info = mi_ranking(inputs.iloc[mi], labels[mi], processor)
                components.append(detail.assign(scope=scope))
                ranks.append(table.assign(scope=scope))
                status["analyses"].append(dict(info, scope=scope))
                orders = {"shap": combined.field.tolist(), "mi": table.sort_values(["rank", "field"]).field.tolist()}
                for selector, fields in orders.items():
                    for fraction in settings.RETENTION_FRACTIONS:
                        count = math.ceil(fraction * len(fields))
                        subsets.extend({"scope": scope, "selector": selector, "retention_fraction": fraction,
                                        "active_original_fields": len(fields), "selected_field_count": count,
                                        "selection_order": index + 1, "field": field}
                                       for index, field in enumerate(fields[:count]))
                status["completed_scopes"].append(scope)
                for filename, table in (("rankings.csv", pd.concat(ranks, ignore_index=True)),
                                        ("common_ranking.csv", pd.concat(common, ignore_index=True)),
                                        ("importance_by_class.csv", pd.DataFrame(class_rows)),
                                        ("selected_features.csv", pd.DataFrame(subsets)),
                                        ("mi_components.csv", pd.concat(components, ignore_index=True))):
                    table.to_csv(staging / filename, index=False)
                    (staging / filename).replace(results / filename)
                print(f"{scope}: saved common SHAP and MI selections at 25% and 50%.", flush=True)
            status["status"] = "complete"
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(results / "selection.json", status, staging)
    print("Training-only feature selection complete; no reduced classifiers fitted and no test scoring.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    select(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

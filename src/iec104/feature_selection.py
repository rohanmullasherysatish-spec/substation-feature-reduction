"""Select common SHAP and MI original fields using training observations only.

Sampling, signed field aggregation and exact mean-rank logic use the shared
selection routines. The common-rank method follows Arreche et al. (2026),
doi:10.1109/TON.2026.3731893, with this study's declared field/class aggregation.
Explanations and MI statistics use each scope's training rows only. SHAP models
use hyperparameters selected by development validation, so their rankings are
conditional on that selected recipe. Test rows and scores are excluded.
"""

import argparse
import importlib.metadata
import json
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

from src.iec104 import settings
from src.feature_selection import (
    selection_recipe as shared_selection_recipe, stratified_rows, shap_ranking, consensus_table,
)
from src.evaluate_validation import full_record, saved_pipeline
from src.checks import require, invalidate_results
from src.train_full_models import atomic_json, load_training_inputs


def selection_recipe():
    """Preserve the declared flow MI representation and the shared SHAP policy."""
    return shared_selection_recipe(
        "one variable per original field; nominal Protocol integer-coded jointly; semantic counts discrete; measurements continuous")


def mi_ranking(frame, labels, processor):
    """Estimate each original variable's association; Protocol is one nominal variable."""
    started = perf_counter()
    fields = processor.active_fields_
    encoded = processor.transform(frame)
    matrix = np.empty((len(frame), len(fields)), dtype=np.float64)
    discrete = np.array([f == "Protocol" or f in settings.INTEGER_FIELDS for f in fields])
    unique = []
    for i, field in enumerate(fields):
        values = processor._categories(frame[field]).ravel() if field == "Protocol" else encoded[:, processor.field_columns([field])[0]]
        levels, codes = np.unique(values, return_inverse=True)
        unique.append(len(levels))
        # Coding avoids treating large count values as continuous target-like floats.
        matrix[:, i] = codes if discrete[i] else values
    variable = np.asarray(unique) > 1
    scores = np.zeros(len(fields))
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=settings.TRAINING_WORKERS), joblib.parallel_backend("threading"):
        warnings.simplefilter("always")
        scores[variable] = mutual_info_classif(matrix[:, variable], labels,
            discrete_features=discrete[variable], n_neighbors=settings.MI_NEIGHBOURS,
            random_state=settings.MI_ESTIMATION_SEED, n_jobs=settings.TRAINING_WORKERS)
    require(np.isfinite(scores).all() and (scores >= 0).all(), "Invalid MI estimates.")
    table = pd.DataFrame({"field": fields, "importance": scores, "rank": rankdata(-scores, method="average"),
                          "method": "mi", "model": "independent", "discrete": discrete,
                          "sample_unique_values": unique, "sample_rows": len(frame)})
    return table, {"method": "mi", "seconds_including_transform": perf_counter() - started,
                   "rows": len(frame), "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}


def select(root=settings.PROJECT_ROOT, overwrite=False):
    root = Path(root).resolve()
    results = root / "results/iec104/feature_selection"
    names = ("rankings.csv", "common_ranking.csv", "importance_by_class.csv", "selected_features.csv", "selection.json")
    require(overwrite or not any((results / n).exists() for n in names), "Selection outputs exist; use --overwrite deliberately.")
    record = full_record(root)
    inputs, labels, scopes, processors = load_training_inputs(root)
    metadata = pd.read_parquet(root / "data/iec104/assignments.parquet")
    metadata = metadata.loc[metadata.partition.eq("development")].reset_index(drop=True)
    samples, sample_positions = [], {}
    plans = [("foreground", settings.SHAP_FOREGROUND_PER_CLASS, settings.SELECTION_SAMPLE_SEED, 0),
             ("background", settings.SHAP_BACKGROUND_PER_CLASS, settings.SELECTION_SAMPLE_SEED, 1),
             ("mi", settings.MI_ROWS_PER_CLASS, settings.MI_SAMPLE_SEED, 2)]
    for index, (scope, mask) in enumerate(scopes.items()):
        for role, cap, seed, role_code in plans:
            selected = np.flatnonzero(mask)[stratified_rows(labels[mask], cap, seed, index, role_code, len(settings.CLASSES))]
            require(mask[selected].all() and len(np.unique(selected)) == len(selected), "Invalid selection sample.")
            sample_positions[scope, role] = selected
            samples.append(metadata.iloc[selected].assign(scope=scope, role=role, development_position=selected))
    sample_table = pd.concat(samples, ignore_index=True)
    local = root / "data/iec104/feature_selection"
    local.mkdir(parents=True, exist_ok=True)
    sample_table.to_parquet(local / "samples.parquet", index=False)
    results.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "recipe": selection_recipe(), "completed_scopes": [], "analyses": [],
              "packages": {n: importlib.metadata.version(n) for n in ("shap", "numpy", "scipy", "scikit-learn")}}
    ranks, common, class_rows, subsets = [], [], [], []
    with TemporaryDirectory(prefix=".selection_iec104_", dir=root) as folder:
        staging = Path(folder)
        invalidate_results(root, 'iec104', 'selection')
        atomic_json(results / "selection.json", status, staging)
        try:
            for scope, mask in scopes.items():
                processor = processors[scope]
                foreground, background, mi = [sample_positions[scope, role] for role in ("foreground", "background", "mi")]
                model_rankings = {}
                for name in settings.CLASSIFIER_PANEL:
                    print(f"{scope}, {name}: explaining {len(foreground)} training rows against {len(background)} background rows.", flush=True)
                    pipeline, _ = saved_pipeline(root, record, name, scope, processor)
                    table, per_class, info = shap_ranking(pipeline, inputs.iloc[foreground], inputs.iloc[background],
                        labels[foreground], processor, local / "shap" / name / f"{scope}.npz", settings.CLASSES)
                    ranks.append(table.assign(scope=scope))
                    class_rows.extend(dict(row, scope=scope) for row in per_class)
                    model_rankings[name] = table
                    status["analyses"].append(dict(info, scope=scope))
                    atomic_json(results / "selection.json", status, staging)
                    print(f"  reconstruction error <= {info['maximum_additivity_absolute_error']:.3g}.", flush=True)
                combined = consensus_table(model_rankings)
                common.append(combined.assign(scope=scope))
                table, info = mi_ranking(inputs.iloc[mi], labels[mi], processor)
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
                                        ("selected_features.csv", pd.DataFrame(subsets))):
                    table.to_csv(staging / filename, index=False)
                    (staging / filename).replace(results / filename)
                print(f"{scope}: saved common SHAP and MI selections at 25% and 50%.", flush=True)
            status["status"] = "complete"
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
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

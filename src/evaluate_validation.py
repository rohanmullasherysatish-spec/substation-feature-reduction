"""Recover selected full-model validation predictions without fitting or test scoring.

The saved predictions link errors to original records and groups. Recalculated
metrics must reproduce the completed training tables before they are published.
"""

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix

from src import settings
from src.modelling import classification_measures
from src.checks import require
from src.train_full_models import atomic_json, load_training_inputs, protocol_classes, training_recipe


def full_record(root, protocol="iec104"):
    """Require completed current full-model training, not an abandoned partial run."""
    record = json.loads((root / f"results/{protocol}/validation/selected_parameters.json").read_text())
    require(record["status"] == "complete", "Full training is incomplete.")
    require(record["recipe"] == training_recipe(protocol), "Full-training settings have changed.")
    for name, version in record["packages"].items():
        require(importlib.metadata.version(name) == version, f"Training dependency changed: {name}")
    return record


def saved_pipeline(root, record, name, scope, processor, protocol="iec104"):
    """Load only the indexed model and check its preprocessing against the current state."""
    path = root / record["selected"][name]["pipelines"][scope]
    require(path == root / f"models/{protocol}/full/{name}/{scope}.joblib" and path.is_file(),
            "Unexpected full-model location.")
    pipeline = joblib.load(path)
    require(pipeline.model_name == name and pipeline.training_scope == scope
            and pipeline.classes_ == protocol_classes(protocol), "Saved model scope or classes differ.")
    require(pipeline.preprocessor.fields == processor.fields
            and pipeline.preprocessor.feature_map_ == processor.feature_map_
            and pd.DataFrame(pipeline.preprocessor.field_records_).equals(pd.DataFrame(processor.field_records_)),
            "Saved model preprocessing differs from current training state.")
    if protocol == "iec61850":
        require(pipeline.preprocessor.states_ == processor.states_
                and pipeline.preprocessor.policy == processor.policy
                and pipeline.preprocessor.settings == processor.settings,
                "Saved packet preprocessing statistics or vocabularies differ.")
    return pipeline, path


def timed_probabilities(pipeline, frame):
    """Repeat one identical prediction task; timing includes preprocessing and batching.

    A short warm-up is excluded. The median is descriptive wall time on this
    machine, not a latency, energy or live flow-extraction benchmark.
    """
    pipeline.predict_proba(frame.iloc[:16])
    times, expected = [], None
    for _ in range(settings.PREDICTION_TIMING_REPEATS):
        started = perf_counter()
        values = pipeline.predict_proba(frame)
        times.append(perf_counter() - started)
        if expected is not None:
            require(np.array_equal(expected.argmax(axis=1), values.argmax(axis=1))
                    and np.allclose(expected, values, atol=1e-12, rtol=1e-12),
                    "Repeated predictions changed materially.")
        expected = values
    return expected, {"pipeline_prediction_seconds": float(np.median(times)),
                      "pipeline_prediction_seconds_min": min(times),
                      "pipeline_prediction_seconds_max": max(times),
                      "prediction_repeats": len(times)}


def prediction_table(metadata, probabilities, model, scope, condition="full", *, class_labels):
    """Keep row identities beside probabilities; identifiers never enter the estimator."""
    classes = tuple(class_labels)
    require(probabilities.shape == (len(metadata), len(classes)), "Prediction table class shape differs.")
    table = metadata[["record_id", "dataset_row", "source_file", "source_row",
                      "group_id", "class_label"]].reset_index(drop=True).copy()
    table = table.rename(columns={"class_label": "true_class"})
    table.insert(0, "condition", condition)
    table.insert(0, "scope", scope)
    table.insert(0, "model", model)
    table["predicted_class"] = np.asarray(classes)[probabilities.argmax(axis=1)]
    for column, label in enumerate(classes):
        table["probability_" + label] = probabilities[:, column]
    return table


def confusion_rows(target, probabilities, model, scope, condition="full", *, class_labels):
    classes = tuple(class_labels)
    matrix = confusion_matrix(target, probabilities.argmax(axis=1), labels=np.arange(len(classes)))
    return [{"model": model, "condition": condition, "scope": scope,
             "true_class": truth, "predicted_class": prediction, "records": int(matrix[i, j])}
            for i, truth in enumerate(classes) for j, prediction in enumerate(classes)]


def recover(root=settings.PROJECT_ROOT, overwrite=False, protocol="iec104"):
    root = Path(root).resolve()
    classes = protocol_classes(protocol)
    results = root / f"results/{protocol}/validation"
    destinations = [root / f"data/{protocol}/full_validation_predictions.parquet"]
    destinations += [results / name for name in ("full_selected_metrics.csv", "full_selected_class_metrics.csv",
                                                 "full_confusion_matrices.csv", "full_errors.csv", "prediction_recovery.json")]
    require(overwrite or not any(p.exists() for p in destinations), "Recovery outputs exist; use --overwrite deliberately.")
    record = full_record(root, protocol)
    inputs, labels, scopes, processors = load_training_inputs(root, protocol)
    metadata = pd.read_parquet(root / f"data/{protocol}/assignments.parquet")
    metadata = metadata.loc[metadata.partition.eq("development")].reset_index(drop=True)
    require(np.array_equal(labels, pd.Categorical(metadata.class_label, categories=classes).codes),
            "Validation metadata or labels are out of order.")
    training = pd.read_csv(results / "full_models.csv")
    metrics, class_metrics, confusions, errors = [], [], [], []
    with TemporaryDirectory(prefix=f".validation_{protocol}_", dir=root) as folder:
        staging = Path(folder)
        with ExitStack() as stack:
            writer = None
            for name in (*settings.CLASSIFIER_PANEL, "dummy"):
                combined = np.empty((len(inputs), len(classes)), dtype=np.float64)
                visited = np.zeros(len(inputs), dtype=bool)
                for scope in scopes:
                    if scope == "development":
                        continue
                    evaluation = ~scopes[scope]
                    require(not visited[evaluation].any(), "A development record was evaluated twice.")
                    pipeline, path = saved_pipeline(root, record, name, scope, processors[scope], protocol)
                    values, timing = timed_probabilities(pipeline, inputs.loc[evaluation])
                    overall, per_class = classification_measures(labels[evaluation], values, classes)
                    old = training.loc[(training.model == name) & (training.scope == scope)
                                       & (training.candidate_index == record["selected"][name]["candidate_index"])]
                    require(len(old) == 1, "Selected training row is absent or duplicated.")
                    for key, value in overall.items():
                        require(np.isclose(value, old.iloc[0]["validation_" + key], atol=1e-12, rtol=1e-12),
                                f"Recovered {key} differs from training: {name}, {scope}")
                    metrics.append({"model": name, "condition": "full", "scope": scope, **overall, **timing,
                                    "active_original_fields": len(pipeline.preprocessor.active_fields_),
                                    "encoded_columns": pipeline.preprocessor.n_features_out_, "model_bytes": path.stat().st_size})
                    class_metrics.extend({"model": name, "condition": "full", "scope": scope, **row} for row in per_class)
                    confusions.extend(confusion_rows(labels[evaluation], values, name, scope, class_labels=classes))
                    table = prediction_table(metadata.loc[evaluation], values, name, scope, class_labels=classes)
                    arrow = pa.Table.from_pandas(table, preserve_index=False)
                    if writer is None:
                        writer = stack.enter_context(pq.ParquetWriter(staging / destinations[0].name, arrow.schema))
                    writer.write_table(arrow)
                    errors.append(table.loc[table.true_class.ne(table.predicted_class),
                                            [c for c in table if not c.startswith("probability_")]])
                    combined[evaluation], visited[evaluation] = values, True
                    print(f"{name}, {scope}: {overall['errors']} errors; saved metrics reproduced.", flush=True)
                require(visited.all(), "Some development records lack a held-fold prediction.")
                overall, per_class = classification_measures(labels, combined, classes)
                metrics.append({"model": name, "condition": "full", "scope": "pooled_validation", **overall})
                class_metrics.extend({"model": name, "condition": "full", "scope": "pooled_validation", **row} for row in per_class)
                confusions.extend(confusion_rows(labels, combined, name, "pooled_validation", class_labels=classes))
        tables = [pd.DataFrame(metrics), pd.DataFrame(class_metrics), pd.DataFrame(confusions), pd.concat(errors, ignore_index=True)]
        for target, table in zip(destinations[1:-1], tables):
            table.to_csv(staging / target.name, index=False)
        atomic_json(staging / destinations[-1].name,
                    {"status": "complete", "models": list(record["selected"]), "validation_records_per_model": len(inputs),
                     "predictions": len(inputs) * len(record["selected"]), "all_training_metrics_reproduced": True,
                     "fitting_performed": False, "test_scoring": False,
                     "pooled_interpretation": "one held-fold prediction per development row; differs from mean fold macro F1",
                     "timing": "three repeated full-pipeline predictions after a 16-row warm-up; descriptive wall time"}, staging)
        for target in destinations:
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / target.name).replace(target)
    print(f"Saved {len(inputs) * len(record['selected']):,} validation predictions; no models fitted and no test scoring.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), default="iec104")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    recover(overwrite=args.overwrite, protocol=args.protocol)


if __name__ == "__main__":
    main()

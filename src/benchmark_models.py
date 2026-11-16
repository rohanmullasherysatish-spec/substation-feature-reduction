"""Benchmark saved pipelines on one fixed prepared population without fitting.

Conditions are interleaved in a fixed random order. Loading, an initial full-task
warm-up and probability verification are outside the timed prediction windows.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from src import settings
from src.checks import require, validate_evaluation, invalidate_results
from src.evaluate_validation import full_record
from src.train_full_models import atomic_json, protocol_classes


def benchmark(root=settings.PROJECT_ROOT, protocol="iec104", repeats=10, seed=50, overwrite=False):
    root = Path(root).resolve()
    require(repeats >= 2, "At least two timing rounds are required.")
    classes = protocol_classes(protocol)
    full_record(root, protocol)
    results = root / f"results/{protocol}/sensitivity/benchmark"
    names = ("prediction_times.csv", "prediction_summary.csv", "benchmark.json")
    require(overwrite or not any((results / n).exists() for n in names),
            "Benchmark outputs exist; use --overwrite deliberately.")
    if protocol == "iec104":
        from src.iec104.partition import read_observations
        observations = read_observations(root)
    else:
        from src.iec61850.partition import read_observations
        observations, _ = read_observations(root)
    assignments = pd.read_parquet(root / f"data/{protocol}/assignments.parquet")
    identities = ["record_id", "dataset_row", "source_file", "source_row", "class_label"]
    require(observations[identities].equals(assignments[identities]), "Benchmark observation identities differ.")
    metadata = assignments.loc[assignments.partition.eq("test")].reset_index(drop=True)
    test = observations.loc[assignments.partition.eq("test")].reset_index(drop=True)
    entries, prediction_sources, evaluation_inputs = {}, {}, {}
    for suffix in ("", "sensitivity/"):
        path = root / f"results/{protocol}/{suffix}test/evaluation.json"
        if not path.exists():
            continue
        selector = "mi_matched" if suffix else None
        record, evaluation_inputs[selector or "main"] = validate_evaluation(root, protocol, selector)
        require(record["status"] == "complete" and record["test_records"] == len(test),
                "Evaluation is incomplete or its population differs.")
        for entry in record["completed_conditions"]:
            key = entry["model"], entry["condition"]
            if key in entries:
                require(entries[key] == entry["pipeline"], "Reference pipeline paths disagree.")
            else:
                entries[key] = entry["pipeline"]
                prediction_sources[key] = root / f"data/{protocol}/{suffix}test_predictions.parquet"
    require(entries, "No completed evaluation pipelines are available.")
    expected_count = 36 if any(c.startswith("mi_matched_") for _, c in entries) else 26
    require(len(entries) == expected_count, "Saved benchmark condition coverage differs.")
    pipelines, frames, expected = {}, {}, {}
    probability_columns = ["probability_" + label for label in classes]
    for key, relative_path in entries.items():
        name, condition = key
        path = root / relative_path
        folder = root / f"models/{protocol}/full/{name}" if condition == "full" else root / f"models/{protocol}/reduced/{condition}/{name}"
        require(path == folder / "development.joblib" and path.resolve() == path, "Unexpected benchmark pipeline path.")
        pipeline = joblib.load(path)
        require(pipeline.model_name == name and pipeline.training_scope == "development"
                and pipeline.classes_ == classes, "Benchmark model scope or classes differ.")
        fields = tuple(pipeline.preprocessor.fields)
        if fields not in frames:
            frames[fields] = test.loc[:, list(fields)]
        saved = pd.read_parquet(prediction_sources[key],
            filters=[("model", "=", name), ("condition", "=", condition)],
            columns=["record_id", "true_class", *probability_columns])
        require(np.array_equal(saved.record_id, metadata.record_id)
                and np.array_equal(saved.true_class, metadata.class_label), "Saved benchmark prediction identities differ.")
        expected[key] = saved[probability_columns].to_numpy(dtype=float)
        pipelines[key] = pipeline
    del observations, assignments
    results.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "protocol": protocol, "started_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(test), "conditions": len(entries), "repeats": repeats, "order_seed": seed,
        "workers": settings.TRAINING_WORKERS, "prediction_batch_rows": settings.PREDICTION_BATCH_ROWS,
        "warmup": "one complete untimed prediction per pipeline before timed rounds",
        "timed_work": "saved pipeline prediction on prepared inputs, including transformation, batching and probabilities",
        "excluded_work": "loading, warmup, verification, capture, flow completion and history construction",
        "order": "independently shuffled condition order each round with one seeded generator",
        "interpretation": "descriptive warm-process local wall times; no fitting, energy or live detection latency measurement",
        "machine": platform.machine(), "system": platform.platform(), "python": sys.version.split()[0],
        "system_load": "external background load not controlled", "rounds_completed": 0,
        "evaluation_inputs": evaluation_inputs,
        "pipelines": [{"model": m, "condition": c, "path": p} for (m, c), p in entries.items()]}
    rows = []

    def verify(key, values):
        require(np.array_equal(values.argmax(axis=1), expected[key].argmax(axis=1))
                and np.allclose(values, expected[key], atol=1e-12, rtol=1e-12),
                f"Benchmark predictions differ from saved evaluation: {key}")

    with TemporaryDirectory(prefix=f".test_{protocol}_", dir=root) as folder:
        staging = Path(folder)
        invalidate_results(root, protocol, "benchmark")
        atomic_json(results / "benchmark.json", status, staging)
        try:
            with threadpool_limits(limits=settings.TRAINING_WORKERS):
                status["numerical_libraries"] = threadpool_info()
                for key, pipeline in pipelines.items():
                    verify(key, pipeline.predict_proba(frames[tuple(pipeline.preprocessor.fields)]))
                keys = list(entries)
                rng = np.random.default_rng(seed)
                for repeat in range(repeats):
                    for position, index in enumerate(rng.permutation(len(keys))):
                        key = keys[index]
                        pipeline = pipelines[key]
                        frame = frames[tuple(pipeline.preprocessor.fields)]
                        started = perf_counter()
                        values = pipeline.predict_proba(frame)
                        seconds = perf_counter() - started
                        verify(key, values)
                        rows.append({"model": key[0], "condition": key[1], "round": repeat + 1,
                            "order_in_round": position + 1, "records": len(frame), "seconds": seconds})
                    status["rounds_completed"] = repeat + 1
                    pd.DataFrame(rows).to_csv(staging / "prediction_times.csv", index=False)
                    (staging / "prediction_times.csv").replace(results / "prediction_times.csv")
                    atomic_json(results / "benchmark.json", status, staging)
                    print(f"Prediction benchmark round {repeat+1}/{repeats}: {len(keys)} pipelines; saved predictions reproduced.", flush=True)
            timings = pd.DataFrame(rows)
            summary = timings.groupby(["model", "condition"]).seconds.agg(
                median_seconds="median", minimum_seconds="min", maximum_seconds="max",
                q25_seconds=lambda s: s.quantile(.25), q75_seconds=lambda s: s.quantile(.75), repeats="size").reset_index()
            refs = summary.loc[summary.condition.eq("full")].set_index("model").median_seconds
            reference_rounds = timings.loc[timings.condition.eq("full")].set_index(["model", "round"]).seconds
            ratios = timings.copy()
            ratios["ratio"] = [row.seconds / reference_rounds.loc[row.model, row.round] for row in ratios.itertuples()]
            paired = ratios.groupby(["model", "condition"]).ratio.agg(
                paired_ratio_median="median", paired_ratio_q25=lambda s: s.quantile(.25),
                paired_ratio_q75=lambda s: s.quantile(.75)).reset_index()
            summary = summary.merge(paired, on=["model", "condition"], validate="one_to_one")
            summary["reduced_full_median_ratio"] = summary.median_seconds / summary.model.map(refs)
            summary["prediction_saving_percent"] = 100 * (1 - summary.reduced_full_median_ratio)
            summary["records"] = len(test)
            summary.to_csv(staging / "prediction_summary.csv", index=False)
            (staging / "prediction_summary.csv").replace(results / "prediction_summary.csv")
            status.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                          probabilities_and_classes_reproduced=True)
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(results / "benchmark.json", status, staging)
    print(summary[["model", "condition", "median_seconds", "prediction_saving_percent"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    benchmark(protocol=args.protocol, repeats=args.repeats, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

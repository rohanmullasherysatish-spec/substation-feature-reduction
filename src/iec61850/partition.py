"""Allocate the prepared packet dependency groups without fitting any model."""

import argparse
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.iec61850 import settings as cfg
from src.checks import require, invalidate_results
from src.partitioning import grouped_assignments, support_table, validate_assignments


def allocation_recipe():
    return {"classes": list(cfg.CLASSES), "outer_folds": cfg.OUTER_FOLDS,
            "test_fold": cfg.TEST_FOLD, "outer_seed": cfg.OUTER_SEED,
            "validation_folds": cfg.VALIDATION_FOLDS, "validation_seed": cfg.VALIDATION_SEED,
            "maximum_test_fraction_deviation": cfg.MAX_TEST_FRACTION_DEVIATION,
            "grouping_dtype": cfg.GROUPING_DTYPE,
            "grouping": "prepared whole-block components connected by equal full numeric/history vectors"}


def read_observations(root):
    """Reject stale preparation, altered identities or incomplete class coverage."""
    root = Path(root)
    summary = json.loads((root/"results/iec61850/preparation/preparation.json").read_text())
    features = json.loads((root/"data/iec61850/features.json").read_text())
    require(summary["status"] == "complete" and summary["recipe"] == cfg.preparation_recipe(),
            "Packet preparation is incomplete or its declared settings changed.")
    require(features["classes"] == list(cfg.CLASSES), "Prepared class order differs.")
    data = pd.read_parquet(root/"data/iec61850/observations.parquet")
    require(list(data.columns) == features["metadata_fields"]+features["predictor_fields"],
            "Prepared packet columns differ from the saved field map.")
    require(len(data) == summary["prepared_records"] and data.record_id.is_unique
            and np.array_equal(data.dataset_row, np.arange(len(data))), "Packet identities or size differ.")
    require(set(data.class_label) == set(cfg.CLASSES), "Prepared packet class coverage differs.")
    require(data.groupby("record_block_id").group_id.nunique().eq(1).all()
            and data.groupby("measurement_group_id").group_id.nunique().eq(1).all(),
            "Prepared dependencies do not stay together.")
    return data, features


def read_development(root):
    """Join the fixed assignments, excluding test rows from downstream fitting."""
    root = Path(root)
    data, features = read_observations(root)
    record = json.loads((root/"results/iec61850/partitioning/partitioning.json").read_text())
    require(record["status"] == "complete" and record["recipe"] == allocation_recipe(),
            "Packet allocation is incomplete or its settings changed.")
    assignments = pd.read_parquet(root/"data/iec61850/assignments.parquet")
    keys = ["record_id", "dataset_row", "source_file", "source_row", "class_label", "group_id"]
    require(data[keys].equals(assignments[keys]), "Saved packet allocation identities or groups differ.")
    validate_assignments(assignments, cfg)
    expected = support_table(assignments, cfg)
    actual = pd.read_csv(root/"results/iec61850/partitioning/support.csv")
    pd.testing.assert_frame_equal(expected, actual)
    mask = assignments.partition.eq("development")
    return data.loc[mask].reset_index(drop=True), assignments.loc[mask].reset_index(drop=True), features


def partition(root=cfg.PROJECT_ROOT, overwrite=False):
    root = Path(root).resolve()
    results = root/"results/iec61850/partitioning"
    outputs = {"assignments.parquet": root/"data/iec61850/assignments.parquet",
               "partitioning.json": results/"partitioning.json", "support.csv": results/"support.csv"}
    for target in outputs.values():
        require(target.resolve() == target and (not target.exists() or target.is_file()),
                f"Output is redirected or not a file: {target}")
    require(overwrite or not any(p.exists() for p in outputs.values()),
            "Allocation outputs exist; deliberate replacement requires --overwrite.")
    data, _ = read_observations(root)
    assignments = grouped_assignments(data, groups=data.group_id, config=cfg)
    require(np.array_equal(assignments.group_id, data.group_id), "Prepared dependency groups changed.")
    support = support_table(assignments, cfg)
    totals = assignments.groupby("partition").agg(records=("record_id", "size"), groups=("group_id", "nunique"))
    record = {"status": "complete", "recipe": allocation_recipe(), "records": len(data),
              "allocation": {name: {k: int(v) for k,v in row.items()}
                             for name,row in totals.to_dict("index").items()},
              "test_fraction": float(assignments.partition.eq("test").mean()),
              "outer_group_overlap": 0, "validation_group_overlap": 0,
              "all_classes_present_in_every_training_and_evaluation_population": True,
              "models_fitted": 0,
              "scope": "Within-collection source-block/measurement separation; not independent captures.",
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow")}}
    with TemporaryDirectory(prefix=".partition_packets_", dir=root) as temporary:
        stage = Path(temporary)
        invalidate_results(root, "iec61850", "partition")
        assignments.to_parquet(stage/"assignments.parquet", index=False, compression="zstd")
        pd.testing.assert_frame_equal(assignments, pd.read_parquet(stage/"assignments.parquet"), check_exact=True)
        support.to_csv(stage/"support.csv", index=False)
        (stage/"partitioning.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        for name,target in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage/name).replace(target)
    print(totals.to_string(), flush=True)
    print(support.loc[support.class_label.eq("ptp_attack")].to_string(index=False), flush=True)
    print("All five classes present in every fitting/evaluation population; whole dependency groups remain separate.")
    print("Saved data/iec61850/assignments.parquet and results/iec61850/partitioning/ tables; no models fitted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    partition(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

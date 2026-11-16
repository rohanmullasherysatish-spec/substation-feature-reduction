"""Assign whole IEC 104 measurement groups to development/test and validation folds.

This operation allocates records only. It does not fit transformations or models,
calculate feature importance, or assess detection performance.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.iec104 import settings
from src.checks import require, invalidate_results
from src.partitioning import grouped_assignments, support_table, validate_assignments


def read_observations(root):
    """Check that the prepared table matches the declared representation."""
    summary_path = root / "results/iec104/preparation/preparation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = pd.read_parquet(root / "data/iec104/observations.parquet")
    require(list(data.columns) == list(settings.PREDICTOR_FIELDS + settings.METADATA_FIELDS),
            "Prepared columns differ from settings; review preparation before partitioning.")
    require(summary["eligible_predictor_fields"] == list(settings.PREDICTOR_FIELDS)
            and summary["excluded_fields"] == settings.EXCLUDED_FIELDS,
            "Saved preparation field rules differ from settings.")
    require(set(summary.get("negative_to_missing_rule", {})) == set(settings.NEGATIVE_TO_MISSING_FIELDS),
            "Saved preparation cleaning rules differ from settings.")
    require(len(data) == summary["retained_records"] and len(data) > 0,
            "Prepared row count differs from its summary.")
    require(data.record_id.is_unique and data.dataset_row.is_unique,
            "Prepared record identities are repeated.")
    require(set(data.class_label) == set(settings.CLASSES), "Prepared class coverage differs.")
    require(data.record_id.notna().all() and data.class_label.notna().all(), "Missing identity or target.")
    for field in settings.NEGATIVE_TO_MISSING_FIELDS:
        require(not data[field].lt(0).any(), f"Uncorrected negative interval: {field}")
    return data


def measurement_groups(data):
    """Group numeric inputs equal at the declared precision, including missing markers."""
    fields = list(settings.GROUPING_FIELDS)
    require(bool(fields) and set(fields).issubset(settings.PREDICTOR_FIELDS), "Invalid grouping fields.")
    require(settings.GROUPING_DTYPE in {"float32", "float64"}, "Unsupported grouping precision.")
    # Cast only a temporary key. The prepared values and model-input values stay intact.
    # Reject overflow rather than grouping unrelated finite values as infinities.
    with np.errstate(over="ignore", invalid="ignore"):
        values = data[fields].to_numpy(dtype=settings.GROUPING_DTYPE, na_value=np.nan)
    require(not np.isinf(values).any(), "Nonfinite or overflowing numeric grouping values.")
    key = pd.DataFrame(values, columns=fields, index=data.index)
    # Integer identifiers follow first occurrence in the fixed prepared-row order.
    # No label, protocol code, source identifier or fitted transformation enters the key.
    groups = key.groupby(fields, sort=False, dropna=False, observed=True).ngroup().astype("int64")
    require(groups.nunique() == len(key.drop_duplicates()),
            "Group count does not match measurement equality at the declared precision.")
    return groups


def partition(root=settings.PROJECT_ROOT, overwrite=False):
    root = Path(root).resolve()
    results = root / "results/iec104/partitioning"
    outputs = {"assignments.parquet": root / "data/iec104/assignments.parquet",
               "partitioning.json": results / "partitioning.json", "support.csv": results / "support.csv"}
    for target in outputs.values():
        require(target.resolve() == target, f"Output path must not redirect through a symlink: {target}")
        require(not target.exists() or target.is_file(), f"Output is not a regular file: {target}")
    require(overwrite or not any(target.exists() for target in outputs.values()),
            "Partition outputs already exist; deliberate replacement requires --overwrite.")
    data = read_observations(root)
    assignments = grouped_assignments(data, measurement_groups(data), settings)
    require(assignments.record_id.equals(data.record_id), "Partitioning changed record identities or order.")
    support = support_table(assignments, settings)
    totals = assignments.groupby("partition").agg(records=("record_id", "size"), groups=("group_id", "nunique"))
    group_sizes = assignments.group_id.value_counts()
    labels_per_group = assignments.groupby("group_id").class_label.nunique()
    dev = assignments.loc[assignments.partition.eq("development")]
    summary = {
        "protocol": "iec104", "records": len(assignments), "measurement_groups": len(group_sizes),
        "grouping_fields": list(settings.GROUPING_FIELDS),
        "grouping_dtype": settings.GROUPING_DTYPE,
        "group_definition": (f"Equality of all eligible unscaled numeric measurements after {settings.GROUPING_DTYPE} "
                             "conversion, including matching missing-value markers; no fitted preprocessing or feature selection."),
        "nominal_fields_omitted_from_grouping": [f for f in settings.PREDICTOR_FIELDS if f not in settings.GROUPING_FIELDS],
        "observations_in_repeated_vectors": int(group_sizes[group_sizes.gt(1)].sum()),
        "largest_group_records": int(group_sizes.max()),
        "mixed_label_groups": int(labels_per_group.gt(1).sum()),
        "allocation": {name: {key: int(value) for key, value in row.items()} for name, row in totals.to_dict("index").items()},
        "outer_folds": settings.OUTER_FOLDS, "test_fold": settings.TEST_FOLD, "outer_seed": settings.OUTER_SEED,
        "test_fraction": float(assignments.partition.eq("test").mean()),
        "maximum_test_fraction_deviation": settings.MAX_TEST_FRACTION_DEVIATION,
        "validation_folds": settings.VALIDATION_FOLDS, "validation_seed": settings.VALIDATION_SEED,
        "fold_sizes": [{"fold": fold, "training_records": int(dev.validation_fold.ne(fold).sum()),
                        "validation_records": int(dev.validation_fold.eq(fold).sum())}
                       for fold in range(settings.VALIDATION_FOLDS)],
        "outer_group_overlap": 0, "validation_group_overlap": 0,
        "all_classes_present_in_every_training_and_evaluation_population": True,
        "models_fitted": 0,
        "scope": "Within-collection measurement-group separation; not independent-capture or unseen-attack evaluation.",
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow")},
    }
    with TemporaryDirectory(prefix=".partition_iec104_", dir=root) as temporary:
        staging = Path(temporary)
        assignments.to_parquet(staging / "assignments.parquet", index=False, compression="zstd")
        restored = pd.read_parquet(staging / "assignments.parquet")
        pd.testing.assert_frame_equal(assignments, restored, check_exact=True)
        validate_assignments(restored, settings)
        support.to_csv(staging / "support.csv", index=False)
        (staging / "partitioning.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        invalidate_results(root, 'iec104', 'partition')
        for name, target in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(target)
    print(f"Allocated {len(assignments):,} observations in {len(group_sizes):,} measurement groups.")
    print(totals.to_string())
    print("Test allocation: " + f"{summary['test_fraction']:.2%}")
    print(pd.DataFrame(summary["fold_sizes"]).to_string(index=False))
    print("All eight classes present in each fitting/evaluation population; no group crossing detected.")
    print("Saved data/iec104/assignments.parquet and results/iec104/partitioning/ outputs.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace the three partition outputs.")
    args = parser.parse_args()
    try:
        partition(overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Partitioning stopped: {error}\n")


if __name__ == "__main__":
    main()

"""Validate IEC 104 CSVs and save prepared observations and numerical summaries.

Raw files are read only. Preparation performs deterministic parsing and duplicate
handling; fitted preprocessing, data partitioning and modelling occur separately.
"""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
from src.checks import invalidate_results
import platform
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

from src.iec104 import settings


class PreparationError(ValueError):
    """A source or output does not satisfy the declared preparation rules."""


def require(condition, message):
    """Stop with an informative error instead of silently accepting bad data."""
    if not condition:
        raise PreparationError(message)


def headers(path):
    """Read the supplied field list, removing its formatting commas and BOM."""
    columns = [
        line.strip().removesuffix(",").strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    require(len(columns) == len(set(columns)), "Repeated source-header field names")
    return columns


def normalise_batch(rows, columns, source_file, first_row, expected_raw_label):
    """Parse numeric fields without learning medians, scales or vocabularies."""
    frame = pd.DataFrame(rows, columns=columns, dtype="string")
    # Retain the original annotation; trim only the copy used for label mapping.
    raw_label = frame["Label"].copy()
    labels = raw_label.str.strip()
    bad_labels = ~labels.eq(expected_raw_label) | ~labels.isin(settings.RAW_LABEL_MAP)
    require(not bad_labels.any(), f"Unexpected label in {source_file}, data records "
            f"{(np.flatnonzero(bad_labels.to_numpy())[:5] + first_row).tolist()}")
    observations = {}
    for field in columns:
        kind = settings.field_kind(field)
        if kind in {"text", "label"}:
            continue
        text = frame[field].str.strip()
        missing = text.str.casefold().isin(settings.MISSING_NUMERIC_TOKENS)
        # Coercion identifies invalid tokens; the next check rejects them explicitly.
        values = pd.to_numeric(text.mask(missing), errors="coerce")
        invalid = values.isna() & ~missing
        require(not invalid.any(), f"Invalid number in {source_file}, field {field}, data records "
                f"{(np.flatnonzero(invalid.to_numpy())[:5] + first_row).tolist()}")
        array = values.to_numpy(dtype="float64", na_value=np.nan)
        positive, negative = np.isposinf(array), np.isneginf(array)
        # Record infinities and defer filling missing values to training-only fitting.
        values = values.mask(positive | negative)
        # Apply the declared field-specific rule without deleting the observation.
        invalid_negative = (
            values.lt(0).fillna(False)
            if field in settings.NEGATIVE_TO_MISSING_FIELDS
            else pd.Series(False, index=values.index)
        )
        values = values.mask(invalid_negative)
        if kind in {"categorical_integer", "integer_count"}:
            require(values.dropna().mod(1).eq(0).all(),
                    f"Nonintegral value in {source_file}, field {field}")
            try:
                # Nullable integers preserve missing cells without inventing values.
                values = values.astype("Int64")
            except (TypeError, ValueError, OverflowError) as error:
                raise PreparationError(f"Integer out of range in {source_file}: {field}") from error
        else:
            values = values.astype("Float64")
        if field in settings.PORT_FIELDS:
            require(values.dropna().between(0, 65535).all(), f"Invalid port in {source_file}: {field}")
        if field == "Protocol":
            require(values.dropna().between(0, 255).all(), f"Invalid protocol code in {source_file}")
        frame[field] = values
        observations[field] = {
            "missing_tokens": int(missing.sum()),
            "positive_infinity": int(positive.sum()),
            "negative_infinity": int(negative.sum()),
            "negative_to_missing": int(invalid_negative.sum()),
        }
    frame = frame.drop(columns="Label")
    frame["raw_label"] = raw_label
    frame["class_label"] = labels.map(settings.RAW_LABEL_MAP).astype("string")
    frame["source_file"] = pd.Series(source_file, index=frame.index, dtype="string")
    # Count CSV data records after the header, including records with quoted newlines.
    frame["source_row"] = np.arange(first_row, first_row + len(frame), dtype=np.int64)
    frame["record_id"] = pd.Series(
        [f"{source_file}::{row}" for row in frame.source_row], dtype="string"
    )
    return frame, observations


def read_sources(source):
    """Validate the declared collection and combine batches in source order."""
    columns = headers(source / "headers_iec104_all.txt")
    require(tuple(columns) == settings.SOURCE_COLUMNS, "Complete source schema differs from settings")
    actual = sorted(path.relative_to(source).as_posix() for path in source.rglob("*.csv"))
    require(actual == [path for path, _ in settings.SOURCES], "Source CSV inventory differs from settings")
    frames, inventory, observed = [], [], {}
    for relative, expected_label in settings.SOURCES:
        print(f"Reading {relative}", flush=True)
        count, batch = 0, []

        def append_batch():
            frame, counts = normalise_batch(
                batch, columns, relative, count - len(batch) + 1, expected_label
            )
            frames.append(frame)
            for field, numbers in counts.items():
                totals = observed.setdefault(field, dict.fromkeys(numbers, 0))
                for key, value in numbers.items():
                    totals[key] += value

        # csv.reader handles quoting; strict width checks prevent column misalignment.
        with (source / relative).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            require(next(reader, None) == columns, f"CSV header differs: {relative}")
            for values in reader:
                count += 1
                require(len(values) == len(columns), f"CSV width differs: {relative}, data record {count}")
                batch.append(values)
                if len(batch) == settings.CSV_ROWS_PER_BATCH:
                    append_batch()
                    batch = []
            if batch:
                append_batch()
        require(count > 0, f"Empty source: {relative}")
        inventory.append({"source_file": relative, "raw_label": expected_label,
                          "class_label": settings.RAW_LABEL_MAP[expected_label], "rows_read": count})
        print(f"  {count:,} data records parsed", flush=True)
    # Global duplicate comparison needs the complete flow collection in memory.
    parsed = pd.concat(frames, ignore_index=True)
    parsed["dataset_row"] = np.arange(len(parsed), dtype=np.int64)
    require(parsed.record_id.is_unique, "Repeated source record identities")
    require(set(parsed.class_label) == set(settings.CLASSES), "Prepared class coverage differs")
    return parsed, pd.DataFrame(inventory), observed


def duplicate_analysis(data):
    """Keep the first exact parsed copy, preserving every conflicting-label row.

    Equality uses all original non-target fields, including the excluded fields.
    These temporary equality groups are not the modelling dependency groups.
    """
    original_fields = [field for field in settings.SOURCE_COLUMNS if field != "Label"]
    # Include missing values when determining equality; do not compare labels yet.
    vectors = data.groupby(original_fields, sort=False, dropna=False, observed=True).ngroup()
    label_counts = data.class_label.groupby(vectors, sort=False).nunique()
    conflicts = label_counts.index[label_counts.gt(1)]
    conflicting = vectors.isin(conflicts)
    # Raw and clean labels must both agree before a row can be an expendable copy.
    first_record = data.record_id.groupby(
        [vectors, data.raw_label, data.class_label], sort=False, dropna=False
    ).transform("first")
    # Conflicting labels remain visible rather than being replaced by a majority label.
    removed = data.record_id.ne(first_record) & ~conflicting
    lineage = data.loc[removed, ["record_id", "source_file", "source_row", "raw_label", "class_label"]].copy()
    lineage["kept_record_id"] = first_record.loc[removed].to_numpy()
    retained = data.loc[~removed].copy()
    require(not conflicting.loc[removed].any(), "A conflicting-label row would be removed")
    return retained, lineage, {
        "duplicate_copies_removed": int(removed.sum()),
        "original_vector_groups_with_conflicting_labels": int(len(conflicts)),
        "rows_in_conflicting_original_vectors": int(conflicting.sum()),
    }


def field_review(parsed, retained, observed):
    """Describe field quality without using global statistics to select inputs."""
    records = []
    for field in settings.SOURCE_COLUMNS:
        column = "raw_label" if field == "Label" else field
        before, after = parsed[column], retained[column]
        kind = settings.field_kind(field)
        if kind in {"text", "label"}:
            before = before.mask(before.str.strip().eq(""))
            after = after.mask(after.str.strip().eq(""))
        role = "target" if field == "Label" else ("excluded" if field in settings.EXCLUDED_FIELDS else "predictor")
        numeric = kind not in {"text", "label"}
        finite = after.dropna()
        records.append({
            "field": field, "role": role, "kind": kind, "reason": settings.field_reason(field),
            "dtype": str(before.dtype), "missing_read": int(before.isna().sum()),
            "missing_retained": int(after.isna().sum()),
            "unique_retained": int(after.nunique(dropna=True)),
            "constant_or_all_missing_retained": bool(after.nunique(dropna=True) <= 1),
            "minimum_retained": float(finite.min()) if numeric and len(finite) else None,
            "maximum_retained": float(finite.max()) if numeric and len(finite) else None,
            **observed.get(field, {}),
        })
    return pd.DataFrame(records)


def prepare(root=settings.PROJECT_ROOT, overwrite=False):
    """Prepare observations and save six outputs under the active project root."""
    root = Path(root).resolve()
    data_dir = root / "data/iec104"
    results_dir = root / "results/iec104/preparation"
    outputs = {
        "observations.parquet": data_dir / "observations.parquet",
        **{name: results_dir / name for name in (
            "preparation.json", "sources.csv", "classes.csv", "fields.csv", "duplicates.csv"
        )},
    }
    # Prevent redirected output paths and require an explicit choice to replace files.
    for target in outputs.values():
        require(target.resolve() == target, f"Output path must not redirect through a symlink: {target}")
        require(not target.exists() or target.is_file(), f"Output is not a regular file: {target}")
    existing = [str(path.relative_to(root)) for path in outputs.values() if path.exists()]
    require(not existing or overwrite, "Preparation outputs already exist: " + ", ".join(existing)
            + ". Use --overwrite only when deliberately replacing these preparation outputs.")

    started = time.perf_counter()
    parsed, sources, observed = read_sources(root / "Datasets/iec104")
    retained, duplicates, duplicate_stats = duplicate_analysis(parsed)
    # Remove excluded fields only after the original-field duplicate check is complete.
    observations = retained[list(settings.PREDICTOR_FIELDS + settings.METADATA_FIELDS)].reset_index(drop=True)
    require(observations.record_id.is_unique, "Prepared record identities are not unique")
    require(set(observations.class_label) == set(settings.CLASSES), "Duplicate handling removed a class")
    # Independent source and class totals must reconcile with the prepared row count.
    kept_per_source = retained.source_file.value_counts()
    sources["rows_retained"] = sources.source_file.map(kept_per_source).fillna(0).astype(int)
    sources["duplicate_copies_removed"] = sources.rows_read - sources.rows_retained
    classes = pd.DataFrame({
        "rows_read": parsed.class_label.value_counts().reindex(settings.CLASSES, fill_value=0),
        "rows_retained": observations.class_label.value_counts().reindex(settings.CLASSES, fill_value=0),
    }).rename_axis("class_label").reset_index()
    classes["duplicate_copies_removed"] = classes.rows_read - classes.rows_retained
    fields = field_review(parsed, retained, observed)
    require(int(sources.rows_retained.sum()) == len(observations) == int(classes.rows_retained.sum()),
            "Source, class and observation totals disagree")
    # Save actual counts and used rules, without assuming any expected result totals.
    summary = {
        "protocol": "iec104", "source_files": len(sources), "raw_records": len(parsed),
        "retained_records": len(observations), "classes": list(settings.CLASSES),
        "eligible_predictor_fields": list(settings.PREDICTOR_FIELDS),
        "excluded_fields": settings.EXCLUDED_FIELDS, "metadata_fields": list(settings.METADATA_FIELDS),
        "duplicate_analysis": duplicate_stats,
        "duplicate_rule": "Exact parsed original non-target fields plus raw and clean labels; keep first in source/record order; retain all rows of conflicting-label vectors.",
        "nonfinite_rule": "Count infinities and replace with missing; imputation must be fitted on training rows later.",
        "negative_to_missing_rule": {
            field: "Replace negative values with missing; retain the observation."
            for field in sorted(settings.NEGATIVE_TO_MISSING_FIELDS)
        },
        "negative_values_converted_to_missing": {
            field: observed[field]["negative_to_missing"]
            for field in sorted(settings.NEGATIVE_TO_MISSING_FIELDS)
        },
        "constant_rule": "Profile only; no global constant or missing-field removal.",
        "partitions_created": False, "models_fitted": 0,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")},
    }
    # Stage complete outputs before replacing the specifically named destination files.
    with TemporaryDirectory(prefix=".prepare_iec104_", dir=root) as temporary:
        staging = Path(temporary)
        observations.to_parquet(staging / "observations.parquet", index=False, compression="zstd")
        # Check that saving and reading the table preserves its values and dtypes.
        pd.testing.assert_frame_equal(observations, pd.read_parquet(staging / "observations.parquet"), check_exact=True)
        for name, table in (("sources.csv", sources), ("classes.csv", classes),
                            ("fields.csv", fields), ("duplicates.csv", duplicates)):
            table.to_csv(staging / name, index=False)
        summary["preparation_seconds"] = time.perf_counter() - started
        (staging / "preparation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        invalidate_results(root, 'iec104', 'preparation')
        for name, target in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(target)
    print(f"Prepared {len(observations):,} observations; removed {len(duplicates):,} exact duplicate copies.")
    print(f"Conflicting original-field vectors: {duplicate_stats['original_vector_groups_with_conflicting_labels']:,}; "
          f"all {duplicate_stats['rows_in_conflicting_original_vectors']:,} associated rows retained.")
    print(f"Eligible original inputs: {len(settings.PREDICTOR_FIELDS)}; constants are retained for training-only review.")
    print("Negative values converted to missing: "
          + str(summary["negative_values_converted_to_missing"]))
    print(classes.to_string(index=False))
    print("Saved data/iec104/observations.parquet and numerical preparation outputs in results/iec104/preparation/.")
    return summary


def main():
    """Expose the preparation operation as a terminal command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the six named preparation outputs.")
    args = parser.parse_args()
    try:
        prepare(overwrite=args.overwrite)
    except (PreparationError, OSError, csv.Error) as error:
        parser.exit(1, f"Preparation stopped: {error}\n")


if __name__ == "__main__":
    main()

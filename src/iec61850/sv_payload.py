"""Strict SV byte parsing and a development-only representation review.

Preserve byte values and ASDU boundaries without assigning channel names,
physical units or a particular measurement profile. The review reads cached
source strings and existing held-fold predictions; it never fits classifiers.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.iec61850 import settings as cfg
from src.iec61850.parsing import parse_value, payload_parts
from src.checks import require
from src.iec61850.partition import read_development


def present(raw):
    return raw is not None and bool(str(raw).strip())


def inspect_sources(root, assignments):
    """Filter cached source strings to development identities before inspection."""
    settings = cfg.field_settings()
    rules = {r["field"]: r for r in settings["field_rules"]}
    limits = {"maximum_characters": settings["maximum_field_characters"],
              "maximum_asdus": settings["maximum_list_items"],
              "maximum_bytes_per_asdu": settings["maximum_field_characters"]//2}
    preparation = json.loads((root/"results/iec61850/preparation/preparation.json").read_text())
    scans = {r["source_file"]: r for r in preparation["completed_sources"]}
    scalar_counts = ("sv.noASDU", "sv.seqASDU")
    repeated = ("sv.ASDU_element", "sv.smpCnt", "sv.svID", "sv.confRev", "sv.smpSynch")
    optional = ("sv.datSet", "sv.smpRate", "sv.smpMod", "sv.meas_value", "sv.meas_quality")
    columns = ["source_file", "source_row", "class_label", "Label", "frame.protocols",
               "sv.seqData", *scalar_counts, *repeated, *optional]
    rows = []
    for item in cfg.sources():
        source = item["path"]
        expected = assignments.loc[assignments.source_file.eq(source)].sort_values("source_row")
        require(len(expected) > 0, f"No development rows for {source}.")
        raw_path = root/"Datasets/iec61850"/source
        current = raw_path.stat()
        require(current.st_size == scans[source]["source_bytes"]
                and current.st_mtime_ns == scans[source]["source_mtime_ns"],
                f"Source changed since preparation: {source}")
        cache = root/".cache/iec61850_samples"/Path(source).stem/"sampled_strings.parquet"
        # Logical filtering occurs before any payload is inspected. Test rows are
        # excluded even though the same cache also holds their sampled strings.
        table = pq.read_table(cache, columns=columns,
            filters=[("source_row", "in", expected.source_row.tolist())]).to_pandas()
        table = table.sort_values("source_row").reset_index(drop=True)
        require(table.source_file.eq(source).all()
                and np.array_equal(table.source_row, expected.source_row)
                and table.class_label.eq(item["class_label"]).all()
                and table.Label.str.strip().eq(item["raw_label"]).all(),
                f"Cached development identities or labels differ: {source}")
        count, malformed, text_max, metadata_missing = 0, 0, 0, 0
        shapes, mismatches, optional_counts = Counter(), Counter(), Counter()
        protocols = Counter(table["frame.protocols"].fillna(""))
        sv_rows = sum(n for stack, n in protocols.items() if "sv" in stack.split(":"))
        for record in table.to_dict("records"):
            for field in optional:
                optional_counts[field] += int(present(record[field]))
            raw = record["sv.seqData"]
            text_max = max(text_max, len(raw) if isinstance(raw, str) else 0)
            try:
                parts = payload_parts(raw, **limits)
            except ValueError:
                malformed += 1
                continue
            if parts is None:
                continue
            count += 1
            # This verifies lossless parsing, including leading zeroes and order.
            require([p.hex() for p in parts] == [p.strip().lower() for p in raw.strip().split(",")],
                    "SV byte roundtrip differs from its source text.")
            shapes["+".join(str(len(p)) for p in parts)] += 1
            absent = False
            for field in scalar_counts:
                if not present(record[field]):
                    absent = True
                else:
                    parsed = parse_value(record[field], rules[field]["kind"], settings)
                    mismatches[field] += int(parsed != len(parts))
            for field in repeated:
                if not present(record[field]):
                    absent = True
                else:
                    # Identity text is used only to check the number of exported
                    # occurrences. No identity becomes a feature or grouping key.
                    mismatches[field] += int(len(record[field].split(",")) != len(parts))
            metadata_missing += int(absent)
        shapes_text = json.dumps(dict(sorted(shapes.items())), separators=(",", ":"))
        rows.append({"source_file": source, "class_label": item["class_label"],
            "development_records": len(table), "sv_protocol_records": sv_rows,
            "payload_present": count, "payload_missing": int(sum(not present(v) for v in table["sv.seqData"])),
            "malformed_payloads": malformed, "asdu_byte_shapes": shapes_text,
            "maximum_text_characters": text_max,
            "payload_rows_with_missing_asdu_metadata": metadata_missing,
            "asdu_metadata_mismatches": int(sum(mismatches.values())),
            "metadata_mismatches_by_field": json.dumps(dict(sorted(mismatches.items())), separators=(",", ":")),
            **{field+"_present": optional_counts[field] for field in optional}})
    return pd.DataFrame(rows), limits


def source_errors(root, assignments):
    """Aggregate saved held-fold errors by source, retaining every declared row."""
    record = json.loads((root/"results/iec61850/validation/prediction_recovery.json").read_text())
    require(record["status"] == "complete" and not record["fitting_performed"]
            and not record["test_scoring"], "Complete validation recovery is required.")
    names = (*cfg.CLASSIFIER_PANEL, "dummy")
    positions = pd.Index(assignments.record_id)
    require(positions.is_unique, "Repeated development identity.")
    visited = {name: np.zeros(len(assignments), dtype=bool) for name in names}
    totals, errors = Counter(), Counter()
    columns = ["model", "condition", "scope", "record_id", "source_file", "group_id", "true_class", "predicted_class"]
    saved = pq.ParquetFile(root/"data/iec61850/full_validation_predictions.parquet")
    for batch in saved.iter_batches(batch_size=65536, columns=columns):
        table = batch.to_pandas()
        index = positions.get_indexer(table.record_id)
        require((index >= 0).all() and table.condition.eq("full").all(),
                "A prediction is outside the declared full development population.")
        expected = assignments.iloc[index]
        require(np.array_equal(table.source_file, expected.source_file)
                and np.array_equal(table.true_class, expected.class_label)
                and np.array_equal(table.group_id, expected.group_id)
                and np.array_equal(table.scope, "fold_"+expected.validation_fold.astype(str))
                and table.predicted_class.isin(cfg.CLASSES).all(), "Validation metadata or class mapping differs.")
        for name, subset in table.groupby("model", sort=False):
            require(name in visited, "Unexpected classifier in validation predictions.")
            idx = positions.get_indexer(subset.record_id)
            require(len(np.unique(idx)) == len(idx) and not visited[name][idx].any(),
                    "Repeated held-fold prediction.")
            visited[name][idx] = True
            for source, label, prediction in subset[["source_file", "true_class", "predicted_class"]].itertuples(index=False, name=None):
                key = (name, source, label)
                totals[key] += 1
                errors[key] += int(label != prediction)
    require(all(value.all() for value in visited.values()), "Incomplete held-fold prediction coverage.")
    return pd.DataFrame([{"model": name, "condition": "full", "scope": "pooled_validation",
        "source_file": source, "class_label": label, "records": n,
        "errors": errors[name,source,label], "error_rate": errors[name,source,label]/n,
        "normal_false_alarm_rate": errors[name,source,label]/n if label == "normal" else None}
        for (name,source,label),n in sorted(totals.items())])


def inspect(root=cfg.PROJECT_ROOT, overwrite=False):
    root = Path(root).resolve()
    folder = root/"results/iec61850/representation"
    files = ("sv_payloads.csv", "source_errors.csv", "representation.json")
    for name in files:
        target = folder/name
        require(target.resolve() == target and (not target.exists() or target.is_file()),
                f"Output is redirected or not a file: {target}")
    require(overwrite or not any((folder/name).exists() for name in files),
            "Representation outputs exist; deliberate replacement requires --overwrite.")
    _, assignments, _ = read_development(root)
    payloads, limits = inspect_sources(root, assignments)
    metrics = source_errors(root, assignments)
    summary = {"status": "complete", "population": "development only",
        "development_records": len(assignments), "parser_limits": limits,
        "payload_records": int(payloads.payload_present.sum()),
        "missing_payload_records": int(payloads.payload_missing.sum()),
        "malformed_payload_records": int(payloads.malformed_payloads.sum()),
        "asdu_metadata_mismatches": int(payloads.asdu_metadata_mismatches.sum()),
        "source_metric_rows": len(metrics), "lossless_byte_roundtrip_checked": True,
        "test_records_inspected": 0, "classifiers_fitted": 0, "new_predictions": 0,
        "diagnostic_changes_representation": False,
        "payload_is_model_input": any(r["field"] == "sv.seqData" and r["role"] == "eligible"
                                      for r in cfg.field_settings()["field_rules"]),
        "physical_channel_schema_verified": False,
        "byte_shapes": "Each shape lists ASDU byte lengths in order; e.g. 32+32 is two 32-byte ASDUs."}
    with TemporaryDirectory(prefix="sv_representation_", dir=root/".cache") as temporary:
        stage = Path(temporary)
        payloads.to_csv(stage/files[0], index=False)
        metrics.to_csv(stage/files[1], index=False)
        (stage/files[2]).write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            (stage/name).replace(folder/name)
    print(payloads[["source_file", "development_records", "payload_present", "asdu_byte_shapes", "malformed_payloads"]].to_string(index=False))
    print("\nNormal-source held-fold false-alarm rates:")
    print(metrics.loc[metrics.class_label.eq("normal"), ["model", "source_file", "records", "errors", "normal_false_alarm_rate"]].to_string(index=False))
    print(f"Saved three numerical review files in {folder.relative_to(root)}; no fitting, test inspection or representation change.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    inspect(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

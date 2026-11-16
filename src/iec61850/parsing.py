"""Typed packet parsing, bounded CSV reading and causal source-block context.

Field semantics are declared in config/iec61850_fields.csv. Nominal values are
not treated as ordered measurements. Parsing and block-history algorithms are
adapted from the reviewed reusable implementation. Raw files are read only.
"""
import csv, io, json, math, re, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from src.checks import require

HISTORY_FIELDS = [
    "history_packet_count", "history_frame_len_mean", "history_frame_len_std",
    "history_frame_len_min", "history_frame_len_max", "history_interarrival_mean",
    "history_interarrival_std", "history_interarrival_sum",
    "history_interarrival_count", "history_same_protocol_fraction",
]

def group_token(value):
    """Float32 equality key only; stored typed measurements retain original precision."""
    if value is None:
        result = None
    elif isinstance(value, list):
        return "[" + ",".join(group_token(v) for v in value) + "]"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.float32(value)
        require(np.isfinite(converted), "Numeric grouping value overflows float32.")
        result = 0.0 if converted == 0 else float(converted)
    return json.dumps(result, separators=(",", ":"), allow_nan=False)

def numeric_codes(array):
    """Canonical measurement codes after declared grouping precision, including missing."""
    encoded = pc.dictionary_encode(array)
    keys = pa.array([group_token(v) for v in encoded.dictionary.to_pylist()])
    mapping = pc.dictionary_encode(keys)
    return pc.fill_null(pc.take(mapping.indices, encoded.indices), -1).to_numpy().astype(np.int32)


def scalar(token, kind, limit):
    if kind == "float64":
        require(
            bool(
                re.fullmatch(
                    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token
                )
            ),
            "Invalid decimal measurement",
        )
        value = float(token)
        require(math.isfinite(value), "Overflow in finite numeric token")
        require(
            value != 0 or Decimal(token).is_zero(),
            "Numeric underflow would erase nonzero data",
        )
        return 0.0 if value == 0 else value
    if kind in ["int64", "decimal_category"]:
        require(
            bool(
                re.fullmatch(
                    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token
                )
            ),
            "Invalid decimal integer text",
        )
        try:
            value = Decimal(token)
        except InvalidOperation as error:
            raise ValueError("Invalid integer") from error
        require(
            value.is_finite()
            and value == value.to_integral_value()
            and abs(value) <= limit,
            "Nonintegral or nonexact integer",
        )
        return str(int(value)) if kind == "decimal_category" else int(value)
    require(
        bool(re.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+", token)), "Invalid hexadecimal code"
    )
    value = int(token, 16)
    require(abs(value) <= limit, "Nonexact hexadecimal integer")
    return (
        ("-" if value < 0 else "") + f"0x{abs(value):x}"
        if kind == "hex_category"
        else value
    )

def payload_parts(raw, maximum_characters=65536, maximum_asdus=64,
                  maximum_bytes_per_asdu=32768):
    """Parse bounded, comma-separated hexadecimal ASDUs without losing zeroes.

    The diagnostic may use broad parsing bounds; preparation supplies the
    narrower declared model-input dimensions. Neither path truncates content.
    Empty source cells mean an absent payload, not an all-zero payload.
    """
    if raw is None or isinstance(raw, str) and not raw.strip():
        return None
    require(isinstance(raw, str) and len(raw) <= maximum_characters,
            "Invalid or excessive SV payload text.")
    parts = [part.strip() for part in raw.strip().split(",")]
    require(1 <= len(parts) <= maximum_asdus, "SV payload exceeds the ASDU bound.")
    decoded = []
    for part in parts:
        require(bool(re.fullmatch(r"[0-9a-fA-F]+", part)) and len(part) % 2 == 0,
                "Malformed SV hexadecimal byte sequence.")
        require(len(part)//2 <= maximum_bytes_per_asdu, "SV ASDU exceeds the byte bound.")
        decoded.append(bytes.fromhex(part))
    return decoded


def parse_value(raw, kind, cfg):
    require(
        isinstance(raw, str) and len(raw) <= cfg["maximum_field_characters"],
        "Invalid or excessive raw field",
    )
    token = raw.strip()
    if not token:
        return None
    if kind == "sv_payload":
        # Preserve byte order, leading zeroes and separate ASDU boundaries.
        # Bounds are declared from development-data format review, not scores.
        bounds = cfg["sv_payload_encoding"]
        decoded = payload_parts(token, cfg["maximum_field_characters"],
                                bounds["maximum_asdus"], bounds["maximum_bytes_per_asdu"])
        return [list(part) for part in decoded]
    if kind == "category":
        return token
    if token.lower() in cfg["missing_tokens"]:
        return None
    if kind.endswith("_list"):
        parts = [part.strip() for part in token.split(",")]
        require(
            len(parts) <= cfg["maximum_list_items"] and all(parts),
            "Invalid or excessive repeated field",
        )
        component = {
            "int_list": "int64",
            "float_list": "float64",
            "hex_list": "hex_integer",
            "decimal_category_list": "decimal_category",
            "hex_category_list": "hex_category",
        }[kind]
        return [
            (
                None
                if part.lower() in cfg["missing_tokens"]
                else scalar(part, component, cfg["exact_float_integer_limit"])
            )
            for part in parts
        ]
    return scalar(token, kind, cfg["exact_float_integer_limit"])

def represent(raw, cfg):
    arrays, names, codes, diagnostics = [], [], {}, []
    for rule in cfg["field_rules"]:
        field, kind = rule["field"], rule["kind"]
        if rule["role"] == "excluded":
            diagnostics.append(
                dict(
                    rule,
                    missing_records=None,
                    distinct_values=None,
                    maximum_observed_list_items=None,
                    nonfinite_tokens_to_missing=None,
                )
            )
            continue
        encoded = pc.dictionary_encode(raw[field].combine_chunks())
        tokens = encoded.dictionary.to_pylist()
        values = []
        nonfinite_counts = []
        for index, token in enumerate(tokens):
            try:
                values.append(parse_value(token, kind, cfg))
            except ValueError as error:
                raise ValueError(field + ": " + str(error)) from error
            parts = token.strip().split(",") if kind.endswith("_list") else [token]
            nonfinite_counts.append(
                sum(
                    bool(part.strip()) and part.strip().lower() in cfg["missing_tokens"]
                    for part in parts
                )
                if kind != "category"
                else 0
            )
        dtype = {
            "sv_payload": pa.list_(pa.list_(pa.uint8())),
            "float64": pa.float64(),
            "int64": pa.int64(),
            "category": pa.string(),
            "hex_category": pa.string(),
            "decimal_category": pa.string(),
            "int_list": pa.list_(pa.int64()),
            "float_list": pa.list_(pa.float64()),
            "hex_list": pa.list_(pa.int64()),
            "decimal_category_list": pa.list_(pa.string()),
            "hex_category_list": pa.list_(pa.string()),
        }[kind]
        parsed = pc.take(pa.array(values, type=dtype), encoded.indices)
        arrays.append(parsed)
        names.append(field)
        if rule["numeric_grouping_field"]:
            canonical = pa.array(
                [group_token(v) for v in values]
            )
            mapping = pc.dictionary_encode(canonical)
            codes[field] = (
                pc.take(mapping.indices, encoded.indices).to_numpy().astype(np.int32)
            )
        distinct = len(set(json.dumps(v, separators=(",", ":")) for v in values))
        diagnostics.append(
            dict(
                rule,
                missing_records=parsed.null_count,
                distinct_values=distinct,
                maximum_observed_list_items=max(
                    [len(v) for v in values if isinstance(v, list)] or [0]
                ),
                nonfinite_tokens_to_missing=int(
                    np.bincount(encoded.indices.to_numpy(), minlength=len(tokens))
                    @ np.asarray(nonfinite_counts)
                ),
            )
        )
    return pa.Table.from_arrays(arrays, names=names), pd.DataFrame(codes), diagnostics

def sampling_plan(records, source_index, settings, seed):
    records, source_index = int(records), int(source_index)
    size, cap = settings["block_records"], settings["maximum_records_per_source"]
    if records < 1 or source_index < 0 or size < 2 or cap < size:
        raise ValueError("Invalid declared source/block budget")
    full, remainder = divmod(records, size)
    if records <= cap:
        chosen = np.arange(full + bool(remainder), dtype=int)
        probability = 1.0
    else:
        number = min(full, cap // size)
        rng = np.random.default_rng(np.random.SeedSequence([seed, source_index]))
        chosen = np.sort(rng.choice(full, number, replace=False))
        probability = number / full
    rows = []
    for block in chosen:
        start, stop = int(block) * size, min((int(block) + 1) * size, records)
        rows.append(
            {
                "source_index": source_index,
                "source_block_index": int(block),
                "source_row_start_zero_based": start,
                "source_row_stop_exclusive": stop,
                "records": stop - start,
                "block_inclusion_probability": probability,
            }
        )
    return rows

def block_history(frame_lengths, frame_intervals, protocol_codes, previous_packets=8):
    """Summarise preceding packets; mask the interval crossing a block boundary."""
    length = np.asarray(frame_lengths, dtype=float)
    interval = np.asarray(frame_intervals, dtype=float).copy()
    codes = list(protocol_codes)
    if (
        length.ndim != 1
        or interval.shape != length.shape
        or len(codes) != len(length)
        or previous_packets < 1
    ):
        raise ValueError("History arrays or history length differ")
    if len(interval):
        interval[0] = np.nan
    if (length[np.isfinite(length)] < 0).any() or (
        interval[np.isfinite(interval)] < 0
    ).any():
        raise ValueError("Negative packet length or relative interval")
    if np.isinf(length).any() or np.isinf(interval).any():
        raise ValueError("Infinite parsed packet measurement")
    result, starts = [], []
    for current in range(len(length)):
        start = max(0, current - previous_packets)
        starts.append(start)
        past_length = length[start:current]
        past_length = past_length[np.isfinite(past_length)]
        past_interval = interval[start + 1 : current]
        past_interval = past_interval[np.isfinite(past_interval)]
        present_codes = [
            code
            for code in codes[start:current]
            if code is not None and str(code) != ""
        ]
        current_code = codes[current]
        same = (
            sum(code == current_code for code in present_codes) / len(present_codes)
            if present_codes and current_code is not None and str(current_code) != ""
            else np.nan
        )
        result.append(
            [
                current - start,
                float(past_length.mean()) if len(past_length) else np.nan,
                float(past_length.std(ddof=0)) if len(past_length) else np.nan,
                float(past_length.min()) if len(past_length) else np.nan,
                float(past_length.max()) if len(past_length) else np.nan,
                float(past_interval.mean()) if len(past_interval) else np.nan,
                float(past_interval.std(ddof=0)) if len(past_interval) else np.nan,
                float(past_interval.sum()) if len(past_interval) else np.nan,
                len(past_interval),
                same,
            ]
        )
    features = pd.DataFrame(result, columns=HISTORY_FIELDS)
    for name in ["history_packet_count", "history_interarrival_count"]:
        features[name] = features[name].astype("int64")
    return {
        "adjusted_frame_time_delta": interval,
        "features": features,
        "context_start_position": np.asarray(starts, dtype=int),
        "context_stop_position_exclusive": np.arange(len(length), dtype=int),
    }

class CountingInput(io.RawIOBase):
    def __init__(self, handle):
        super().__init__()
        self.handle = handle
        self.bytes_read = 0
        self.eof = False

    def readable(self):
        return True

    def read(self, size=-1):
        require(0 <= size <= 64 * 1024**2, "Unbounded/excessive input request")
        data = self.handle.read(size)
        self.bytes_read += len(data)
        if not data and size:
            self.eof = True
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

def scan_source(
    path, item, source_index, all_fields, selected, cfg, directory, progress_seconds=15
):
    before = {"bytes": path.stat().st_size}
    with path.open(newline="", encoding="utf8") as handle:
        require(
            next(csv.reader(handle)) == all_fields,
            "Source header differs: " + path.name,
        )
    plan = sampling_plan(
        item["records"], source_index, cfg["sampling"], cfg["seeds"]["sampling"]
    )
    positions = np.concatenate(
        [
            np.arange(
                row["source_row_start_zero_based"],
                row["source_row_stop_exclusive"],
                dtype=np.int64,
            )
            for row in plan
        ]
    )
    blocks = np.concatenate(
        [
            np.full(row["records"], row["source_block_index"], dtype=np.int64)
            for row in plan
        ]
    )
    samples = []
    count = 0
    clock = time.perf_counter()
    last = clock
    with path.open("rb") as handle:
        stream = CountingInput(handle)
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(
                use_threads=False,
                block_size=8 * 1024**2,
                column_names=all_fields,
                skip_rows=1,
                encoding="utf8",
            ),
            parse_options=pacsv.ParseOptions(
                newlines_in_values=True, ignore_empty_lines=False
            ),
            convert_options=pacsv.ConvertOptions(
                include_columns=selected,
                column_types={name: pa.string() for name in selected},
                strings_can_be_null=False,
                null_values=[],
                check_utf8=True,
            ),
        )
        try:
            for batch in reader:
                label = pc.utf8_trim_whitespace(batch.column(selected.index("Label")))
                require(
                    bool(pc.all(pc.equal(label, item["raw_label"])).as_py()),
                    "Unexpected target label in "
                    + path.name
                    + " near record "
                    + str(count),
                )
                lo, hi = np.searchsorted(positions, [count, count + batch.num_rows])
                if hi > lo:
                    chosen = pa.Table.from_batches([batch]).take(
                        pa.array(positions[lo:hi] - count)
                    )
                    chosen = chosen.append_column(
                        "source_row", pa.array(positions[lo:hi] + 1)
                    )
                    chosen = chosen.append_column(
                        "source_block_index", pa.array(blocks[lo:hi])
                    )
                    samples.append(chosen)
                count += batch.num_rows
                if time.perf_counter() - last >= progress_seconds:
                    print(
                        f'  {count:,}/{item["records"]:,} records; {stream.bytes_read:,}/{before["bytes"]:,} bytes; {time.perf_counter()-clock:.0f}s',
                        flush=True,
                    )
                    last = time.perf_counter()
        finally:
            reader.close()
        require(
            stream.eof and stream.bytes_read == before["bytes"],
            "Incomplete source consumption",
        )
    require(
        count == item["records"], "Fresh record count differs from source inventory"
    )
    table = pa.concat_tables(samples)
    require(
        table.num_rows == len(positions)
        and np.array_equal(table["source_row"].to_numpy(), positions + 1),
        "Block selection/order differs",
    )
    table = table.append_column(
        "source_index", pa.array(np.full(len(positions), source_index, dtype=np.int64))
    )
    table = table.append_column(
        "source_file", pa.array([item["path"]] * len(positions))
    )
    table = table.append_column(
        "scenario", pa.array([item["scenario"]] * len(positions))
    )
    table = table.append_column(
        "class_label", pa.array([item["class_label"]] * len(positions))
    )
    table = table.append_column(
        "block_inclusion_probability",
        pa.array([plan[0]["block_inclusion_probability"]] * len(positions)),
    )
    pq.write_table(
        table,
        directory / "sampled_strings.parquet",
        compression="zstd",
        row_group_size=8192,
    )
    require(
        pq.read_table(directory / "sampled_strings.parquet").equals(table),
        "Source sample roundtrip differs",
    )
    with (directory / "sampling_blocks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan[0]))
        writer.writeheader()
        writer.writerows(plan)
    return {
        "source_index": source_index,
        "source_file": item["path"],
        "class_label": item["class_label"],
        "source_bytes": before["bytes"],
        "fresh_validated_records": count,
        "selected_records": len(positions),
        "selected_source_blocks": len(plan),
        "all_record_widths_and_labels_checked": True,
        "full_block_inclusion_probability": plan[0]["block_inclusion_probability"],
        "unselected_large_source_tail": (
            count % cfg["sampling"]["block_records"]
            if count > cfg["sampling"]["maximum_records_per_source"]
            else 0
        ),
        "elapsed_seconds": time.perf_counter() - clock,
        "sample_parquet_roundtrip_exact": True,
    }

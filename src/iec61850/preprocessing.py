"""Training-only typed packet encoding with maps to original feature fields.

Bounded list/category encoding preserves maps back to each prepared field.
Numeric scaling is model-specific; evaluation rows never change fitted rules.
"""

from collections import Counter
import argparse
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from src.iec61850 import settings as cfg
from src.checks import require, invalidate_results
from src.iec61850.partition import allocation_recipe, read_development


def missing(value):
    return (
        value is None
        or value is pd.NA
        or isinstance(value, (float, np.floating))
        and np.isnan(value)
    )


def token(value):
    return "M" if missing(value) else "V" + str(value)


class PacketPreprocessor:
    def __init__(self, field_rules, settings, maximum_matrix_bytes, expansion_policy):
        self.rules = [dict(r) for r in field_rules if r["role"] == "eligible"]
        self.fields = tuple(r["field"] for r in self.rules)
        self.model_panel = tuple(cfg.CLASSIFIER_PANEL)
        self.standardize_for = tuple(cfg.STANDARDIZE_NUMERIC_FOR)
        self.settings = dict(settings)
        self.policy = dict(expansion_policy)
        self.maximum_matrix_bytes = int(maximum_matrix_bytes)

    def _validate_frame(self, frame):
        require(
            tuple(frame.columns) == self.fields,
            "Predictor interface differs from declared fields; labels and metadata are not accepted",
        )

    def _components(self, series, rule):
        """Expand the fixed schema, independently of observed split sizes or scores."""
        values = series.tolist()
        if rule["kind"] == "sv_payload":
            bounds = self.policy["sv_payload_encoding"]
            asdus, width = bounds["maximum_asdus"], bounds["maximum_bytes_per_asdu"]
            rows = len(values)
            present = np.asarray([not missing(v) for v in values], dtype=np.float64)
            counts = np.zeros(rows, dtype=np.float64)
            lengths = np.zeros((rows, asdus), dtype=np.float64)
            slots = np.full((rows, asdus, width), np.nan, dtype=np.float64)
            for i, payload in enumerate(values):
                if missing(payload):
                    continue
                require(1 <= len(payload) <= asdus, "SV payload exceeds the declared ASDU layout")
                counts[i] = len(payload)
                for a, part in enumerate(payload):
                    require(1 <= len(part) <= width, "SV payload exceeds the declared byte layout")
                    byte_values = np.asarray(part, dtype=np.float64)
                    require(np.isfinite(byte_values).all() and np.equal(byte_values, np.floor(byte_values)).all()
                            and ((byte_values >= 0) & (byte_values <= 255)).all(), "Invalid unsigned payload byte")
                    lengths[i, a] = len(part)
                    slots[i, a, :len(part)] = byte_values
            # Zero length identifies an absent ASDU. Presence/count/lengths
            # distinguish a missing payload from observed all-zero content.
            components = {"present": present, "asdu_count": counts}
            for a in range(asdus):
                components[f"asdu_{a:02d}_length"] = lengths[:, a]
                for b in range(width):
                    components[f"asdu_{a:02d}_byte_{b:02d}"] = slots[:, a, b]
            return components
        if "list" not in rule["kind"]:
            if "category" in rule["kind"]:
                return {"value": np.asarray([token(v) for v in values], dtype=object)}
            return {"value": series.to_numpy(dtype=np.float64, na_value=np.nan)}
        limit = int(rule["list_component_limit"])
        require(
            limit == self.policy["list_component_limit"],
            "List declaration differs from fixed policy",
        )
        rows = len(values)
        present = np.asarray([not missing(v) for v in values], dtype=np.float64)
        lengths = np.asarray(
            [len(v) if not missing(v) else np.nan for v in values], dtype=np.float64
        )
        require(
            not np.any(lengths > limit),
            "Repeated field exceeds fixed list limit; no truncation allowed",
        )
        categorical = "category" in rule["kind"]
        slots = (
            np.full((rows, limit), "M", dtype=object)
            if categorical
            else np.full((rows, limit), np.nan)
        )
        for i, value in enumerate(values):
            if not missing(value):
                slots[i, : len(value)] = (
                    [token(v) for v in value]
                    if categorical
                    else [np.nan if missing(v) else float(v) for v in value]
                )
        return {
            "length": lengths,
            "present": present,
            **{f"item_{j:02d}": slots[:, j] for j in range(limit)},
        }

    @staticmethod
    def _numeric_state(raw, discrete):
        require(not np.isinf(raw).any(), "Infinity reached fitted preprocessing")
        good = raw[np.isfinite(raw)]
        if not len(good):
            return None
        median = float(np.median(good))
        imputed = np.where(np.isnan(raw), median, raw)
        mean, scale = float(imputed.mean()), float(imputed.std())
        require(
            np.isfinite(mean) and np.isfinite(scale),
            "Nonfinite training scaling statistics",
        )
        variable = bool(np.any(imputed != imputed[0]))
        indicator = bool(np.isnan(raw).any() and not np.isnan(raw).all())
        if not variable and not indicator:
            return None
        return {
            "kind": "numeric",
            "median": median,
            "mean": mean,
            "scale": scale or 1.0,
            "value_column": variable,
            "missing_indicator": indicator,
            "mi_discrete": discrete,
        }

    def _component_numeric_state(self, raw, rule, name):
        """Apply the same declared missingness rule during fitting and checking."""
        discrete = name in {"length", "present"} or rule["mi_component_type"] == "discrete"
        state = self._numeric_state(raw, discrete)
        if rule["kind"] == "sv_payload" and "_byte_" in name and state is not None:
            # Presence and ASDU lengths identify absent byte positions already.
            # Keep training-varying values without repeating identical flags.
            state["missing_indicator"] = False
            if not state["value_column"]:
                return None
        return state

    def _vocabulary(self, components, rule):
        if "category" not in rule["kind"]:
            return None
        counts = Counter()
        for name, raw in components.items():
            if name.startswith("item_") or name == "value":
                counts.update(raw.tolist())
        cap = int(self.settings["maximum_categories_per_field"])
        require(cap >= 2, "Category cap must allow retained and pooled categories")
        ordered = sorted(counts, key=lambda value: (-counts[value], value))
        has_missing = "M" in counts
        if len(counts) > cap:
            kept = (["M"] if has_missing else []) + [v for v in ordered if v != "M"][
                : cap - 1 - int(has_missing)
            ]
            pooled = sorted(set(counts) - set(kept))
        else:
            kept, pooled = sorted(counts), []
        return {
            "known": sorted(counts),
            "kept": sorted(kept),
            "pooled": pooled,
            "infrequent_token": "I" if pooled else None,
            "category_count_including_missing": len(counts),
            "maximum_categories_per_field": cap,
        }

    @staticmethod
    def _category_codes(raw, vocabulary):
        kept = set(vocabulary["kept"])
        pooled = vocabulary["infrequent_token"]
        return np.asarray(
            [v if v in kept else pooled if v != "M" and pooled else "U" for v in raw],
            dtype=object,
        )

    def fit(self, frame, record_ids):
        self._validate_frame(frame)
        require(
            len(frame) > 0 and len(record_ids) == len(frame),
            "Invalid training population",
        )
        require(
            len(set(record_ids)) == len(record_ids), "Repeated training record identity"
        )
        self.training_rows_ = len(frame)
        self.category_levels_, self.vocabularies_ = {}, {}
        self.field_records_ = []
        (
            self.states_,
            self.feature_map_,
            self.dropped_fields_,
            self.component_screen_,
        ) = ([], [], [], [])
        for rule in self.rules:
            field = rule["field"]
            components = self._components(frame[field], rule)
            vocabulary = self._vocabulary(components, rule)
            if vocabulary is not None:
                self.vocabularies_[field] = vocabulary
                self.category_levels_[field] = frozenset(vocabulary["known"])
            initial_width = len(self.feature_map_)
            field_state = {
                "field": field,
                "source_kind": rule["kind"],
                "vocabulary": vocabulary,
                "components": [],
            }
            for name, raw in components.items():
                categorical = vocabulary is not None and (
                    name == "value" or name.startswith("item_")
                )
                if categorical:
                    codes = self._category_codes(raw, vocabulary)
                    unique = sorted(set(codes))
                    if len(unique) <= 1:
                        state = None
                    else:
                        columns = sorted(
                            set(unique)
                            | ({"I"} if vocabulary["infrequent_token"] else set())
                        )
                        state = {"kind": "categorical", "categories": columns}
                else:
                    state = self._component_numeric_state(raw, rule, name)
                if state is None:
                    self.component_screen_.append(
                        {
                            "field": field,
                            "component": name,
                            "reason": "all_missing_or_constant_in_training",
                        }
                    )
                    continue
                state["component"] = name
                field_state["components"].append(state)
                if categorical:
                    entries = [
                        {
                            "encoding": "missing_indicator" if v == "M" else "one_hot",
                            "category_token": v,
                            "mi_discrete": True,
                        }
                        for v in state["categories"]
                    ]
                else:
                    entries = []
                    if state["value_column"]:
                        entries.append(
                            {
                                "encoding": "numeric",
                                "mi_discrete": state["mi_discrete"],
                            }
                        )
                    if state["missing_indicator"]:
                        entries.append(
                            {"encoding": "missing_indicator", "mi_discrete": True}
                        )
                for entry in entries:
                    self.feature_map_.append(
                        {
                            "column_index": len(self.feature_map_),
                            "original_field": field,
                            "component": name,
                            **entry,
                        }
                    )
            if field_state["components"]:
                self.states_.append(field_state)
            else:
                all_missing = all(missing(v) for v in frame[field])
                self.dropped_fields_.append(
                    {
                        "field": field,
                        "reason": (
                            "all_missing_in_training"
                            if all_missing
                            else "constant_in_training"
                        ),
                    }
                )
            self.field_records_.append({
                "original_field": field, "kind": rule["kind"],
                "training_rows": len(frame),
                "training_missing": sum(missing(v) for v in frame[field]),
                "status": "active" if field_state["components"] else "dropped",
                "encoded_columns": len(self.feature_map_) - initial_width,
                "reason": "" if field_state["components"] else self.dropped_fields_[-1]["reason"],
                "training_categories": len(vocabulary["known"]) if vocabulary else 0,
                "pooled_training_categories": len(vocabulary["pooled"]) if vocabulary else 0,
            })
        require(bool(self.states_), "No active training fields")
        self.active_fields_ = tuple(s["field"] for s in self.states_)
        self.n_features_out_ = len(self.feature_map_)
        return self

    def transform(self, frame, standardize=False):
        self._validate_frame(frame)
        require(hasattr(self, "states_"), "Preprocessor has not been fitted")
        require(
            len(frame) * self.n_features_out_ * 8 <= self.maximum_matrix_bytes,
            "Requested encoded matrix exceeds the declared memory budget",
        )
        matrix = np.empty((len(frame), self.n_features_out_), dtype=np.float64)
        offset = 0
        rules = {r["field"]: r for r in self.rules}
        for field_state in self.states_:
            parts = self._components(
                frame[field_state["field"]], rules[field_state["field"]]
            )
            for state in field_state["components"]:
                raw = parts[state["component"]]
                if state["kind"] == "categorical":
                    values = self._category_codes(raw, field_state["vocabulary"])
                    for category in state["categories"]:
                        matrix[:, offset] = values == category
                        offset += 1
                else:
                    require(not np.isinf(raw).any(), "Infinity reached transform")
                    absent = np.isnan(raw)
                    if state["value_column"]:
                        values = np.where(absent, state["median"], raw)
                        if standardize:
                            values = (values - state["mean"]) / state["scale"]
                        matrix[:, offset] = values
                        offset += 1
                    if state["missing_indicator"]:
                        matrix[:, offset] = absent
                        offset += 1
        require(offset == self.n_features_out_, "Encoded field-map width differs")
        require(np.isfinite(matrix).all(), "Nonfinite encoded matrix")
        matrix[matrix == 0] = 0.0
        return matrix

    def field_columns(self, selected_fields):
        require(
            len(selected_fields) == len(set(selected_fields)), "Repeated selected field"
        )
        require(
            set(selected_fields).issubset(self.active_fields_),
            "Selected field is not active in training",
        )
        selected = set(selected_fields)
        return [
            e["column_index"]
            for e in self.feature_map_
            if e["original_field"] in selected
        ]

    def transform_for_model(self, frame, model):
        require(model in self.model_panel or model == "dummy", "Unknown classifier.")
        return self.transform(frame, standardize=model in self.standardize_for)

    def unknown_category_counts(self, frame):
        self._validate_frame(frame)
        result = {}
        rules = {r["field"]: r for r in self.rules}
        for field, known in self.category_levels_.items():
            parts = self._components(frame[field], rules[field])
            slots = [v for k,v in parts.items() if k == "value" or k.startswith("item_")]
            result[field] = {
                "unseen_nonmissing_values": sum(sum(v != "M" and v not in known for v in part) for part in slots),
                "missing_values_not_observed_in_training": sum(sum(v == "M" and v not in known for v in part) for part in slots),
            }
        return result

    def validate_training_frame(self, frame):
        """Recalculate fitted statistics on the permitted training population only."""
        self._validate_frame(frame)
        require(len(frame) == self.training_rows_, "Saved packet training size differs.")
        fields = {s["field"]: s for s in self.states_}
        columns = []
        for rule in self.rules:
            field = rule["field"]
            parts = self._components(frame[field], rule)
            vocabulary = self._vocabulary(parts, rule)
            require(vocabulary == self.vocabularies_.get(field), f"Training vocabulary differs: {field}")
            expected = []
            for name,raw in parts.items():
                if vocabulary is not None and (name == "value" or name.startswith("item_")):
                    unique = sorted(set(self._category_codes(raw, vocabulary)))
                    state = None if len(unique) <= 1 else {
                        "kind": "categorical", "categories": sorted(set(unique) | ({"I"} if vocabulary["infrequent_token"] else set()))}
                else:
                    state = self._component_numeric_state(raw, rule, name)
                if state is not None:
                    expected.append(dict(state, component=name))
            require(expected == fields.get(field, {}).get("components", []), f"Training component state differs: {field}")
            if expected:
                columns.extend([field] * sum(len(s["categories"]) if s["kind"] == "categorical"
                                            else int(s["value_column"]) + int(s["missing_indicator"]) for s in expected))
        require(tuple(fields) == self.active_fields_, "Saved packet active-field order differs.")
        require(columns == [c["original_field"] for c in self.feature_map_]
                and len(columns) == self.n_features_out_, "Saved packet component mapping differs.")


def exact_overlap(training, validation, dtype):
    """Compare complete encoded row bytes, without treating hashes as equality.

    This is a diagnostic only: no matching observations are removed or reassigned.
    """
    if not len(validation):
        for start in range(0, len(training), 4096):
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(training[start:start+4096], dtype=dtype)
            require(np.isfinite(values).all(), "Nonfinite training inputs at the diagnostic precision.")
        return None
    matrix = np.ascontiguousarray(np.concatenate([training, validation]).astype(dtype, copy=False))
    require(np.isfinite(matrix).all(), "Nonfinite inputs at the diagnostic precision.")
    matrix[matrix == 0] = 0.0
    rows = matrix.view(np.dtype((np.void, matrix.shape[1]*matrix.dtype.itemsize))).ravel()
    _, inverse = np.unique(rows, return_inverse=True)
    return int(np.isin(inverse[len(training):], np.unique(inverse[:len(training)])).sum())


def bundle_recipe(features):
    return {"predictor_fields": tuple(features["predictor_fields"]),
            "feature_rules": features["feature_rules"], "classes": tuple(cfg.CLASSES),
            "classifier_panel": tuple(cfg.CLASSIFIER_PANEL), "reference_strategy": cfg.REFERENCE_STRATEGY,
            "standardize_numeric_for": tuple(cfg.STANDARDIZE_NUMERIC_FOR),
            "grouping_fields": tuple(features["numeric_grouping_fields"]), "grouping_dtype": cfg.GROUPING_DTYPE,
            "allocation_recipe": allocation_recipe(), "policy": cfg.preprocessing_policy(),
            "maximum_matrix_bytes": cfg.MAXIMUM_ENCODED_MATRIX_BYTES}


def load_training_inputs(root):
    """Require current assignments, recipes and training-derived fitted states."""
    root = Path(root)
    data, assignments, features = read_development(root)
    bundle = joblib.load(root/"models/iec61850/preprocessors.joblib")
    require(bundle["recipe"] == bundle_recipe(features), "Packet preprocessing settings changed.")
    for name,version in bundle["packages"].items():
        require(importlib.metadata.version(name) == version, f"Packet preprocessing package changed: {name}")
    inputs = data[features["predictor_fields"]]
    labels = pd.Categorical(data.class_label, categories=cfg.CLASSES).codes.astype(np.int64)
    scopes = {f"fold_{fold}": assignments.validation_fold.ne(fold).to_numpy()
              for fold in range(cfg.VALIDATION_FOLDS)}
    scopes["development"] = np.ones(len(data), dtype=bool)
    processors = bundle["preprocessors"]
    require(set(processors) == set(scopes), "Packet preprocessing scope coverage differs.")
    for scope,mask in scopes.items():
        require(set(labels[mask]) == set(range(len(cfg.CLASSES))), f"Missing training class: {scope}")
        if scope != "development":
            require(set(labels[~mask]) == set(range(len(cfg.CLASSES))), f"Missing validation class: {scope}")
        processor = processors[scope]
        require(processor.fields == tuple(features["predictor_fields"])
                and processor.model_panel == tuple(cfg.CLASSIFIER_PANEL)
                and processor.standardize_for == tuple(cfg.STANDARDIZE_NUMERIC_FOR), "Packet predictor interface differs.")
        processor.validate_training_frame(inputs.loc[mask])
    return inputs, labels, scopes, processors


def preprocess(root=cfg.PROJECT_ROOT, overwrite=False):
    """Fit four scopes and save readable maps, without transforming test inputs."""
    root = Path(root).resolve()
    results = root/"results/iec61850/preprocessing"
    names = ("fields.csv", "columns.csv", "component_screen.csv", "scopes.csv", "unknown_categories.csv")
    outputs = {name: results/name for name in names}
    outputs["preprocessors.joblib"] = root/"models/iec61850/preprocessors.joblib"
    for target in outputs.values():
        require(target.resolve() == target and (not target.exists() or target.is_file()),
                f"Output is redirected or not a file: {target}")
    require(overwrite or not any(p.exists() for p in outputs.values()),
            "Packet preprocessing outputs exist; deliberate replacement requires --overwrite.")
    data, assignments, features = read_development(root)
    inputs = data[features["predictor_fields"]]
    scopes = {f"fold_{fold}": assignments.validation_fold.ne(fold).to_numpy()
              for fold in range(cfg.VALIDATION_FOLDS)}
    scopes["development"] = np.ones(len(data), dtype=bool)
    processors, field_rows, column_rows, dropped_rows, scope_rows, unknown_rows = {}, [], [], [], [], []
    for scope,mask in scopes.items():
        started = perf_counter()
        training, validation = inputs.loc[mask], inputs.loc[~mask]
        processor = PacketPreprocessor(features["feature_rules"],
            {"maximum_categories_per_field": cfg.MAXIMUM_CATEGORIES_PER_FIELD},
            cfg.MAXIMUM_ENCODED_MATRIX_BYTES, {"list_component_limit": cfg.LIST_COMPONENT_LIMIT,
             "sv_payload_encoding": dict(cfg.SV_PAYLOAD_ENCODING)})
        processor.fit(training, data.loc[mask, "record_id"])
        fit_seconds = perf_counter()-started
        processors[scope] = processor
        field_rows.extend(dict(r, scope=scope) for r in processor.field_records_)
        column_rows.extend(dict(r, scope=scope) for r in processor.feature_map_)
        dropped_rows.extend(dict(r, scope=scope) for r in processor.component_screen_)
        if len(validation):
            unknown_rows.extend(dict(scope=scope, original_field=field, **counts)
                                for field,counts in processor.unknown_category_counts(validation).items())
        print(f"{scope}: {len(training):,} training rows; {len(processor.active_fields_)} active original fields; "
              f"{processor.n_features_out_} encoded columns; fitted state in {fit_seconds:.2f}s.", flush=True)
        for standardize in (False, True):
            train_matrix = processor.transform(training, standardize=standardize)
            valid_matrix = processor.transform(validation, standardize=standardize)
            overlap64 = exact_overlap(train_matrix, valid_matrix, np.float64)
            overlap32 = exact_overlap(train_matrix, valid_matrix, np.float32)
            scope_rows.append({"scope": scope, "numeric_view": "standardised" if standardize else "unscaled",
                "training_rows": len(training), "validation_rows": len(validation),
                "active_original_fields": len(processor.active_fields_), "encoded_columns": processor.n_features_out_,
                "training_matrix_bytes": int(train_matrix.nbytes), "preprocessor_fit_seconds_shared": fit_seconds,
                "validation_rows_matching_training_float64": overlap64,
                "validation_rows_matching_training_float32": overlap32, "float32_nonfinite_rows": 0})
            del train_matrix, valid_matrix
    tables = dict(zip(names, map(pd.DataFrame, (field_rows, column_rows, dropped_rows, scope_rows, unknown_rows))))
    bundle = {"recipe": bundle_recipe(features), "preprocessors": processors,
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "joblib")}}
    with TemporaryDirectory(prefix=".preprocess_packets_", dir=root) as folder:
        stage = Path(folder)
        invalidate_results(root, "iec61850", "preprocessing")
        joblib.dump(bundle, stage/"preprocessors.joblib", compress=3)
        restored = joblib.load(stage/"preprocessors.joblib")
        # Include real development examples from every class, then check both views.
        probe_rows = data.groupby("class_label", sort=False).head(2).index
        probe = inputs.loc[probe_rows]
        for scope,processor in processors.items():
            restored["preprocessors"][scope].validate_training_frame(inputs.loc[scopes[scope]])
            for standardize in (False, True):
                np.testing.assert_array_equal(processor.transform(probe, standardize=standardize),
                    restored["preprocessors"][scope].transform(probe, standardize=standardize))
        for name,table in tables.items():
            table.to_csv(stage/name, index=False)
        for name,target in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage/name).replace(target)
    print("Saved models/iec61850/preprocessors.joblib and five preprocessing CSVs; test inputs were not transformed.", flush=True)
    return tables["scopes.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    preprocess(overwrite=args.overwrite)


if __name__ == "__main__":
    # Keep the importable class name in saved objects rather than __main__.
    from src.iec61850.preprocessing import main as run_cli
    run_cli()

"""Fit IEC 104 preprocessing on fold-training rows and all development rows.

This command saves reusable transformations and numerical descriptions. It does
not train classifiers, tune settings, select features or transform test inputs.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.iec104 import settings
from src.checks import require, invalidate_results
from src.iec104.partition import measurement_groups, read_observations
from src.partitioning import validate_assignments


class FlowPreprocessor:
    """Learn field screening, medians, categories and scales from training only.

    One fitted state supplies unscaled numeric inputs for trees and standardised
    numeric inputs for Logistic Regression. Neither view rescales one-hot values.
    All encoded columns retain their original-field mapping for later analyses.
    """

    def __init__(self, fields=None):
        self.fields = tuple(settings.PREDICTOR_FIELDS if fields is None else fields)
        require(bool(self.fields) and len(set(self.fields)) == len(self.fields),
                "Predictor fields must be nonempty and unique.")
        require(set(self.fields).issubset(settings.PREDICTOR_FIELDS),
                "Only declared predictors are accepted; labels and metadata are excluded.")
        self.maximum_matrix_bytes = settings.MAXIMUM_ENCODED_MATRIX_BYTES
        self.model_panel = tuple(settings.CLASSIFIER_PANEL)
        self.standardize_for = tuple(settings.STANDARDIZE_NUMERIC_FOR)
        self.fitted_ = False

    def _validate_frame(self, frame):
        require(isinstance(frame, pd.DataFrame) and tuple(frame.columns) == self.fields,
                "Input columns or order differ from this preprocessor's predictors.")

    @staticmethod
    def _numeric(series):
        require(pd.api.types.is_numeric_dtype(series.dtype),
                f"A nonnumeric value reached preprocessing: {series.name}")
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        require(not np.isinf(values).any(), f"Infinity reached preprocessing: {series.name}")
        return values

    @classmethod
    def _categories(cls, series):
        values = cls._numeric(series)
        present = values[~np.isnan(values)]
        require(np.equal(present, np.floor(present)).all()
                and ((present >= 0) & (present <= 255)).all(), "Invalid Protocol category.")
        # Zero is a real code. Missingness is separate, not a new numeric code.
        return (series.astype("Int64").astype("string").fillna("__missing__")
                .to_numpy(dtype=object).reshape(-1, 1))

    def fit(self, frame):
        """Fit only the explicitly supplied training predictor table; never labels."""
        self._validate_frame(frame)
        require(len(frame) > 0, "Cannot fit preprocessing without training rows.")
        self.fitted_ = False
        self.training_rows_ = len(frame)
        self.states_, self.feature_map_, self.field_records_ = [], [], []
        self.category_levels_ = {}
        for field in self.fields:
            series = frame[field]
            raw = self._numeric(series)
            missing = np.isnan(raw)
            record = {
                "original_field": field, "kind": settings.field_kind(field),
                "training_rows": len(frame), "training_missing": int(missing.sum()),
                "status": "dropped", "reason": "", "encoded_columns": 0,
                "median": None, "mean": None, "scale": None, "training_categories": "",
            }
            if field == "Protocol":
                categories = self._categories(series)
                levels = sorted(set(categories.ravel()))
                self.category_levels_[field] = frozenset(levels)
                record["training_categories"] = json.dumps(levels)
                if len(levels) <= 1:
                    record["reason"] = "all_missing_in_training" if missing.all() else "constant_in_training"
                    self.field_records_.append(record)
                    continue
                # No pooling, reference-category deletion or evaluation-derived levels.
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                        drop=None, dtype=np.float64)
                encoder.fit(categories)
                components = [{"component": f"category={level}", "encoding": "one_hot",
                               "mi_discrete": True} for level in encoder.categories_[0]]
                state = {"field": field, "kind": "categorical", "encoder": encoder,
                         "width": len(components)}
            else:
                present = raw[~missing]
                if len(present) == 0:
                    record["reason"] = "all_missing_in_training"
                    self.field_records_.append(record)
                    continue
                median = float(np.median(present))
                imputed = np.where(missing, median, raw)
                record["median"] = median
                # Screening after filling catches one observed value plus missing cells.
                # No missing indicators are introduced for the isolated corrected cell.
                if np.equal(imputed, imputed[0]).all():
                    record["reason"] = "constant_after_training_imputation"
                    self.field_records_.append(record)
                    continue
                mean, scale = float(imputed.mean()), float(imputed.std(ddof=0))
                require(np.isfinite(mean) and np.isfinite(scale) and scale > 0,
                        f"Invalid training scale: {field}")
                record.update(mean=mean, scale=scale)
                state = {"field": field, "kind": "numeric", "median": median,
                         "mean": mean, "scale": scale, "width": 1}
                components = [{"component": "value", "encoding": "numeric",
                               "mi_discrete": field in settings.INTEGER_FIELDS}]
            record.update(status="active", reason="", encoded_columns=state["width"])
            self.field_records_.append(record)
            self.states_.append(state)
            for component in components:
                self.feature_map_.append({"column_index": len(self.feature_map_),
                                          "original_field": field, **component})
        require(bool(self.states_), "No active inputs remain in this training population.")
        self.active_fields_ = tuple(state["field"] for state in self.states_)
        self.n_features_out_ = len(self.feature_map_)
        self.fitted_ = True
        return self

    def transform(self, frame, standardize=False):
        """Apply fixed training rules without changing the fitted state."""
        require(self.fitted_, "Fit preprocessing on training rows first.")
        self._validate_frame(frame)
        require(len(frame) * self.n_features_out_ * 8 <= self.maximum_matrix_bytes,
                "Encoded matrix exceeds its memory limit; transform smaller batches.")
        matrix = np.empty((len(frame), self.n_features_out_), dtype=np.float64)
        offset = 0
        for state in self.states_:
            field, width = state["field"], state["width"]
            if state["kind"] == "categorical":
                if len(frame):
                    matrix[:, offset:offset + width] = state["encoder"].transform(self._categories(frame[field]))
            else:
                raw = self._numeric(frame[field])
                values = np.where(np.isnan(raw), state["median"], raw)
                if standardize:
                    values = (values - state["mean"]) / state["scale"]
                matrix[:, offset] = values
            offset += width
        require(np.isfinite(matrix).all(), "Nonfinite transformed inputs.")
        matrix[matrix == 0] = 0.0
        return matrix

    def transform_for_model(self, frame, model):
        """Choose the declared numeric view for a panel member or the reference."""
        require(model in self.model_panel or model == "dummy", "Unknown classifier name.")
        return self.transform(frame, standardize=model in self.standardize_for)

    def field_columns(self, selected_fields):
        """Return all components of selected fields in the fixed encoded order."""
        require(self.fitted_, "Fit preprocessing before selecting columns.")
        selected = tuple(selected_fields)
        require(bool(selected) and len(set(selected)) == len(selected)
                and set(selected).issubset(self.active_fields_), "Invalid selected original fields.")
        return [entry["column_index"] for entry in self.feature_map_
                if entry["original_field"] in selected]

    def unknown_category_counts(self, frame):
        """Count unseen categories, including for Protocol dropped as constant."""
        require(self.fitted_, "Fit preprocessing before reviewing categories.")
        self._validate_frame(frame)
        return {field: int(sum(value not in levels for value in self._categories(frame[field]).ravel()))
                for field, levels in self.category_levels_.items()}

    def validate_training_frame(self, frame):
        """Verify that a saved state belongs to the current training data, without refitting.

        Direct comparisons catch a stale state or swapped folds without creating
        hashes, learning from evaluation rows or silently replacing fitted rules.
        """
        require(self.fitted_, "Preprocessor is not fitted.")
        self._validate_frame(frame)
        require(len(frame) == self.training_rows_, "Saved preprocessor training size differs.")
        states = {state["field"]: state for state in self.states_}
        records = {record["original_field"]: record for record in self.field_records_}
        for field in self.fields:
            raw = self._numeric(frame[field])
            missing = np.isnan(raw)
            require(int(missing.sum()) == records[field]["training_missing"],
                    f"Saved training missing count differs: {field}")
            if field == "Protocol":
                levels = frozenset(self._categories(frame[field]).ravel())
                require(levels == self.category_levels_[field], "Saved Protocol categories differ.")
                active = len(levels) > 1
                if active:
                    require(field in states and tuple(states[field]["encoder"].categories_[0]) == tuple(sorted(levels)),
                            "Saved Protocol encoder differs from its training categories.")
            else:
                present = raw[~missing]
                active = False
                if len(present):
                    median = float(np.median(present))
                    require(median == records[field]["median"], f"Saved training median differs: {field}")
                    imputed = np.where(missing, median, raw)
                    active = not np.equal(imputed, imputed[0]).all()
                    if active:
                        require(median == states.get(field, {}).get("median")
                                and float(imputed.mean()) == states.get(field, {}).get("mean")
                                and float(imputed.std(ddof=0)) == states.get(field, {}).get("scale"),
                                f"Saved training mean or scale differs: {field}")
            require(active == (field in states), f"Saved training screening differs: {field}")
        require(tuple(states) == self.active_fields_, "Saved active-field order differs.")
        require(sum(state["width"] for state in self.states_) == self.n_features_out_ == len(self.feature_map_),
                "Saved encoded width differs from the column map.")


def read_development(root):
    """Check existing allocation against current observations, then return development."""
    data = read_observations(root)
    assignments = pd.read_parquet(root / "data/iec104/assignments.parquet")
    identities = ["record_id", "dataset_row", "source_file", "source_row", "class_label"]
    require(set(identities + ["group_id", "partition", "validation_fold"]) == set(assignments.columns),
            "Unexpected assignment-table columns.")
    require(data[identities].equals(assignments[identities]),
            "Prepared identities, labels or order differ from the saved assignments.")
    allocation = json.loads((root / "results/iec104/partitioning/partitioning.json").read_text())
    for key, expected in {
        "grouping_fields": list(settings.GROUPING_FIELDS), "outer_folds": settings.OUTER_FOLDS,
        "grouping_dtype": settings.GROUPING_DTYPE,
        "test_fold": settings.TEST_FOLD, "outer_seed": settings.OUTER_SEED,
        "validation_folds": settings.VALIDATION_FOLDS, "validation_seed": settings.VALIDATION_SEED,
    }.items():
        require(allocation.get(key) == expected, f"Saved allocation differs from settings: {key}")
    validate_assignments(assignments, settings)
    for name in ("development", "test"):
        population = assignments.loc[assignments.partition.eq(name)]
        require(set(population.class_label) == set(settings.CLASSES),
                f"A class is absent from the saved {name} population.")
        require(len(population) == allocation["allocation"][name]["records"]
                and population.group_id.nunique() == allocation["allocation"][name]["groups"],
                f"Saved {name} allocation counts differ from its numerical summary.")
    # A deliberate preparation edit must not silently invalidate the saved groups.
    current_groups = measurement_groups(data)
    require(np.array_equal(current_groups.to_numpy(), assignments.group_id.to_numpy()),
            "Current measurement groups differ; review preparation and allocation before fitting.")
    development = assignments.partition.eq("development")
    return (data.loc[development].reset_index(drop=True),
            assignments.loc[development].reset_index(drop=True))


def load_training_inputs(root):
    """Check saved flow transformations against the fixed development population."""
    development, assignments = read_development(root)
    bundle = joblib.load(root / "models/iec104/preprocessors.joblib")
    for key, expected in {
        "predictor_fields": tuple(settings.PREDICTOR_FIELDS),
        "classes": tuple(settings.CLASSES),
        "classifier_panel": tuple(settings.CLASSIFIER_PANEL),
        "reference_strategy": settings.REFERENCE_STRATEGY,
        "standardize_numeric_for": tuple(settings.STANDARDIZE_NUMERIC_FOR),
        "grouping_fields": tuple(settings.GROUPING_FIELDS),
        "grouping_dtype": settings.GROUPING_DTYPE,
    }.items():
        require(bundle.get(key) == expected, f"Saved preprocessing setting differs: {key}")
    for name, version in bundle["packages"].items():
        require(importlib.metadata.version(name) == version,
                f"Preprocessing package version changed: {name}; review compatibility first.")
    processors = bundle["preprocessors"]
    scopes = {f"fold_{fold}": assignments.validation_fold.ne(fold).to_numpy()
              for fold in range(settings.VALIDATION_FOLDS)}
    scopes["development"] = np.ones(len(development), dtype=bool)
    require(set(processors) == set(scopes), "Saved preprocessing scopes differ.")
    inputs = development[list(settings.PREDICTOR_FIELDS)]
    labels = pd.Categorical(development.class_label, categories=settings.CLASSES).codes.astype(np.int64)
    require((labels >= 0).all(), "Unexpected development labels.")
    for scope, mask in scopes.items():
        processor = processors[scope]
        require(set(labels[mask]) == set(range(len(settings.CLASSES))),
                f"A class is absent from training: {scope}")
        if scope != "development":
            require(set(labels[~mask]) == set(range(len(settings.CLASSES))),
                    f"A class is absent from validation: {scope}")
        require(processor.fields == tuple(settings.PREDICTOR_FIELDS)
                and processor.model_panel == tuple(settings.CLASSIFIER_PANEL)
                and processor.standardize_for == tuple(settings.STANDARDIZE_NUMERIC_FOR),
                f"Preprocessor interface differs: {scope}")
        processor.validate_training_frame(inputs.loc[mask])
    return inputs, labels, scopes, processors


def matching_rows(training, evaluation):
    """Count exact encoded evaluation vectors also present in training; do not filter."""
    if len(evaluation) == 0:
        return 0
    combined = pd.DataFrame(np.concatenate([training, evaluation], axis=0))
    groups = combined.groupby(list(combined.columns), sort=False, dropna=False).ngroup()
    return int(groups.iloc[len(training):].isin(groups.iloc[:len(training)]).sum())


def check_scope(processor, training, validation, scope):
    """Describe both input views, leaving validation outside every fitted statistic."""
    rows = []
    unknown = processor.unknown_category_counts(validation).get("Protocol", 0)
    for standardize in (False, True):
        train_matrix = processor.transform(training, standardize=standardize)
        valid_matrix = processor.transform(validation, standardize=standardize)
        overlap64 = matching_rows(train_matrix, valid_matrix) if len(validation) else None
        # This is a precision diagnostic, not a claim to reproduce every tree backend.
        with np.errstate(over="ignore", invalid="ignore"):
            train32, valid32 = train_matrix.astype(np.float32), valid_matrix.astype(np.float32)
        nonfinite32 = int((~np.isfinite(train32)).any(axis=1).sum()
                          + (~np.isfinite(valid32)).any(axis=1).sum())
        overlap32 = (matching_rows(train32, valid32)
                     if len(validation) and nonfinite32 == 0 else None)
        rows.append({"scope": scope, "numeric_view": "standardised" if standardize else "unscaled",
                     "training_rows": len(training), "validation_rows": len(validation),
                     "active_original_fields": len(processor.active_fields_),
                     "encoded_columns": processor.n_features_out_,
                     "validation_unknown_protocol": unknown,
                     "validation_rows_matching_training_float64": overlap64,
                     "validation_rows_matching_training_float32": overlap32,
                     "float32_nonfinite_rows": nonfinite32})
    return rows


def preprocess(root=settings.PROJECT_ROOT, overwrite=False):
    """Create four training-fitted preprocessors and three consolidated CSV tables."""
    root = Path(root).resolve()
    results = root / "results/iec104/preprocessing"
    outputs = {"preprocessors.joblib": root / "models/iec104/preprocessors.joblib",
               **{name: results / name for name in ("fields.csv", "columns.csv", "scopes.csv")}}
    for target in outputs.values():
        require(target.resolve() == target and (not target.exists() or target.is_file()),
                f"Output is redirected or is not a regular file: {target}")
    require(overwrite or not any(path.exists() for path in outputs.values()),
            "Preprocessing outputs already exist; deliberate replacement requires --overwrite.")
    require(set(settings.STANDARDIZE_NUMERIC_FOR).issubset(settings.CLASSIFIER_PANEL),
            "Scaling policy includes a classifier outside the panel.")
    development, assignments = read_development(root)
    inputs = development[list(settings.PREDICTOR_FIELDS)]
    processors, field_tables, column_tables, scope_rows = {}, [], [], []
    scopes = [(f"fold_{fold}", assignments.validation_fold.ne(fold))
              for fold in range(settings.VALIDATION_FOLDS)]
    scopes.append(("development", pd.Series(True, index=inputs.index)))
    for scope, training_mask in scopes:
        training, validation = inputs.loc[training_mask], inputs.loc[~training_mask]
        require(set(development.loc[training_mask, "class_label"]) == set(settings.CLASSES),
                f"A class is absent from training: {scope}")
        if scope != "development":
            require(set(development.loc[~training_mask, "class_label"]) == set(settings.CLASSES),
                    f"A class is absent from validation: {scope}")
        processor = FlowPreprocessor().fit(training)
        processors[scope] = processor
        field_tables.append(pd.DataFrame(processor.field_records_).assign(scope=scope))
        column_tables.append(pd.DataFrame(processor.feature_map_).assign(scope=scope))
        scope_rows.extend(check_scope(processor, training, validation, scope))
        print(f"{scope}: {len(training):,} training rows; {len(processor.active_fields_)} active original inputs; "
              f"{processor.n_features_out_} encoded columns.", flush=True)
    tables = {"fields.csv": pd.concat(field_tables, ignore_index=True),
              "columns.csv": pd.concat(column_tables, ignore_index=True),
              "scopes.csv": pd.DataFrame(scope_rows)}
    bundle = {"preprocessors": processors, "predictor_fields": tuple(settings.PREDICTOR_FIELDS),
              "grouping_fields": tuple(settings.GROUPING_FIELDS), "grouping_dtype": settings.GROUPING_DTYPE,
              "classes": tuple(settings.CLASSES), "classifier_panel": tuple(settings.CLASSIFIER_PANEL),
              "reference_strategy": settings.REFERENCE_STRATEGY,
              "standardize_numeric_for": tuple(settings.STANDARDIZE_NUMERIC_FOR),
              "rules": {"numeric_imputation": "training_median", "numeric_screening": "after_imputation",
                        "protocol": "training_one_hot_unknown_all_zero", "missing_indicators": False,
                        "numeric_output_dtype": "float64"},
              "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name)
                           for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib")}}
    with TemporaryDirectory(prefix=".preprocess_iec104_", dir=root) as temporary:
        staging = Path(temporary)
        joblib.dump(bundle, staging / "preprocessors.joblib", compress=3)
        restored = joblib.load(staging / "preprocessors.joblib")
        # Check persistence with development examples only; no evaluation inputs fit state.
        probe = inputs.iloc[:32]
        for scope, processor in processors.items():
            for standardize in (False, True):
                np.testing.assert_array_equal(processor.transform(probe, standardize=standardize),
                    restored["preprocessors"][scope].transform(probe, standardize=standardize))
        for name, table in tables.items():
            table.to_csv(staging / name, index=False)
        invalidate_results(root, 'iec104', 'preprocessing')
        for name, target in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(target)
    print("Saved models/iec104/preprocessors.joblib and results/iec104/preprocessing/ tables.")
    print("No classifiers fitted. Test inputs were excluded from fitting and transformation checks.")
    return tables["scopes.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace the four preprocessing outputs.")
    args = parser.parse_args()
    try:
        preprocess(overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preprocessing stopped: {error}\n")


if __name__ == "__main__":
    # Use the importable class path so saved objects load from other project scripts.
    from src.iec104.preprocessing import main as run_cli
    run_cli()

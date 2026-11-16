"""Packet representation choices, independent of observed classifier scores."""

import csv
from src.settings import PROJECT_ROOT
from src.settings import (
    OUTER_FOLDS, TEST_FOLD, OUTER_SEED, VALIDATION_FOLDS, VALIDATION_SEED,
    MAX_TEST_FRACTION_DEVIATION, CLASSIFIER_PANEL, STANDARDIZE_NUMERIC_FOR,
    REFERENCE_STRATEGY,
)

CLASSES = ("flood", "fuzzy", "normal", "ptp_attack", "replay")
SAMPLING = {"maximum_records_per_source": 25_000, "block_records": 16}
SAMPLING_SEED = 49
PREVIOUS_PACKETS = 8
GROUPING_DTYPE = "float32"
MAXIMUM_ENCODED_MATRIX_BYTES = 2 * 1024**3
MAXIMUM_CATEGORIES_PER_FIELD = 128
LIST_COMPONENT_LIMIT = 64
# A development-only format review found two 32-byte ASDUs per present payload.
# Reject a larger layout instead of silently dropping bytes or expanding on test.
SV_PAYLOAD_ENCODING = {
    "maximum_asdus": 2,
    "maximum_bytes_per_asdu": 32,
    "representation": "ordered_unsigned_bytes_by_asdu_and_position",
    "feature_unit": "sv.seqData",
    "missingness": "payload_presence_and_asdu_lengths; no_repeated_byte_indicators",
    "physical_measurement_decoder": False,
}


def preprocessing_policy():
    """Bound list/category expansion without using evaluation populations."""
    return {
        "list_component_limit": LIST_COMPONENT_LIMIT,
        "maximum_categories_per_field": MAXIMUM_CATEGORIES_PER_FIELD,
        "numeric_imputation": "training_median",
        "screening": "training_only_all_missing_or_constant_components",
        "missingness": "retain_training_varying_indicators; no_new_evaluation_columns",
        "category_pooling": "training_frequency; lexical_ties; missing_and_infrequent_in_cap",
        "unseen_categories": "existing_infrequent_bucket_otherwise_all_zero",
        "standardize_numeric_for": list(STANDARDIZE_NUMERIC_FOR),
        "output_dtype": "float64",
        "sv_payload_encoding": dict(SV_PAYLOAD_ENCODING),
    }


def sources():
    with (PROJECT_ROOT / "config/iec61850_sources.csv").open(newline="") as handle:
        values = list(csv.DictReader(handle))
    for row in values:
        row["records"] = int(row.pop("expected_records"))
    return values


def field_settings():
    with (PROJECT_ROOT / "config/iec61850_fields.csv").open(newline="") as handle:
        rules = list(csv.DictReader(handle))
    for rule in rules:
        rule["numeric_grouping_field"] = rule["numeric_grouping_field"] == "True"
        rule["list_component_limit"] = int(rule["list_component_limit"]) if rule["list_component_limit"] else None
        rule["feature_unit"] = rule["field"]
    return {"maximum_field_characters": 65536, "maximum_list_items": 64,
            "exact_float_integer_limit": 2**53, "sv_payload_encoding": dict(SV_PAYLOAD_ENCODING),
            "missing_tokens": ["", "nan", "+nan", "-nan", "inf", "+inf", "-inf",
                               "infinity", "+infinity", "-infinity"], "field_rules": rules}


def preparation_recipe():
    return {"classes": list(CLASSES), "sampling": SAMPLING, "sampling_seed": SAMPLING_SEED,
            "previous_packets": PREVIOUS_PACKETS, "grouping_dtype": GROUPING_DTYPE,
            "field_settings": field_settings(), "sources": sources(),
            "target": "provided capture-scenario label, not independently verified packet maliciousness",
            "population": "unweighted source-capped block sample; not natural deployment prevalence",
            "history": "preceding packets within the same source block; first inter-block interval masked",
            "predictor_pool": "author-listed candidates minus declared exclusions, plus ten causal history units",
            "duplicate_policy": "retain packet occurrences and conflicting labels; link block dependencies",
            "grouping": "whole source blocks connected by equal base numeric/history vectors at float32; payload does not weaken groups; labels/identities excluded"}

"""IEC 104 source schema, cleaning rules and model-input representation."""

from src.settings import (
    PROJECT_ROOT, OUTER_FOLDS, TEST_FOLD, OUTER_SEED, VALIDATION_FOLDS,
    VALIDATION_SEED, MAX_TEST_FRACTION_DEVIATION, CLASSIFIER_PANEL,
    REFERENCE_STRATEGY, STANDARDIZE_NUMERIC_FOR, SHAP_FOREGROUND_PER_CLASS,
    SHAP_BACKGROUND_PER_CLASS, SELECTION_SAMPLE_SEED, SHAP_ADDITIVITY_ATOL,
    SHAP_ADDITIVITY_RTOL, MI_ROWS_PER_CLASS, MI_NEIGHBOURS, MI_SAMPLE_SEED,
    MI_ESTIMATION_SEED, RETENTION_FRACTIONS, REDUCED_PARAMETER_POLICY,
    PREDICTION_TIMING_REPEATS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    MIN_BOOTSTRAP_GROUPS_PER_CLASS, ESTIMATOR_SEED, TRAINING_WORKERS,
)

# Bound one dense flow matrix on the local machine.
MAXIMUM_ENCODED_MATRIX_BYTES = 512 * 1024 * 1024

# Bound each CSV conversion batch; the parsed flow table is assembled in memory.
CSV_ROWS_PER_BATCH = 50_000
# Recognise explicit missing tokens. Zero remains a valid observed value.
MISSING_NUMERIC_TOKENS = frozenset({"", "nan", "na", "n/a", "null", "none"})
# Negative elapsed intervals in this field are unavailable measurements, not zero.
# Keep the observation and defer filling the missing cell to training-only fitting.
NEGATIVE_TO_MISSING_FIELDS = frozenset({"Flow IAT Min"})

# Group at the declared model-input precision without changing saved observations.
# Use all eligible numeric inputs before scaling, imputation or feature reduction.
GROUPING_DTYPE = "float32"

# Preserve the supplied annotation separately and use these consistent target names.
RAW_LABEL_MAP = {
    "dosattack": "dos",
    "floodattack": "flood",
    "fuzzyattack": "fuzzy",
    "iec104starvationattack": "iec104_starvation",
    "mitmattack": "mitm",
    "ntpddosattack": "ntp_ddos",
    "portscanattack": "port_scan",
    "attackfree": "normal",
}
# A fixed order prevents later metric/probability columns from changing order.
CLASSES = tuple(sorted(RAW_LABEL_MAP.values()))

# Paths are relative to Datasets/iec104; processing follows this sorted order.
SOURCES = (
    ("attack-data/capture104-dosattack.csv", "dosattack"),
    ("attack-data/capture104-floodattack.csv", "floodattack"),
    ("attack-data/capture104-fuzzyattack.csv", "fuzzyattack"),
    ("attack-data/capture104-iec104starvationattack.csv", "iec104starvationattack"),
    ("attack-data/capture104-mitmattack.csv", "mitmattack"),
    ("attack-data/capture104-ntpddosattack.csv", "ntpddosattack"),
    ("attack-data/capture104-portscanattack.csv", "portscanattack"),
    ("attack-free-data/capture104-attackfree.csv", "attackfree"),
)

# Exact names and order prevent shifted columns or unexpected fields being accepted.
SOURCE_COLUMNS = (
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
)

# These fields are read for source validation and exact-duplicate comparison,
# then omitted from the prepared modelling table. They remain in the raw CSVs.
EXCLUDED_FIELDS = {
    "Flow ID": "Composite endpoint/connection identifier; outside the chosen representation.",
    "Src IP": "Device/environment address; outside the chosen representation.",
    "Dst IP": "Device/environment address; outside the chosen representation.",
    "Timestamp": "Absolute capture time; relative duration and interval measurements remain.",
    "Src Port": "Port-independent representation; sacrifices connection/service context.",
    "Dst Port": "Port-independent representation; sacrifices connection/service context.",
}
TEXT_FIELDS = frozenset({"Flow ID", "Src IP", "Dst IP", "Timestamp"})
# Validate port values while reading, even though ports are excluded from inputs.
PORT_FIELDS = frozenset({"Src Port", "Dst Port"})
# Semantic counts remain integers; other numeric measurements use floating point.
INTEGER_FIELDS = frozenset({
    "Total Fwd Packet", "Total Bwd packets", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Subflow Fwd Packets", "Subflow Bwd Packets", "Fwd Act Data Pkts",
})
# Use this explicit allowlist later; selecting every non-label column is unsafe.
PREDICTOR_FIELDS = tuple(
    field for field in SOURCE_COLUMNS if field != "Label" and field not in EXCLUDED_FIELDS
)
# These columns locate observations and hold labels; none is a model input.
METADATA_FIELDS = (
    "record_id", "source_file", "source_row", "dataset_row", "raw_label", "class_label",
)


def field_kind(field):
    """Distinguish nominal codes, integer counts and continuous measurements."""
    if field == "Label":
        return "label"
    if field in TEXT_FIELDS:
        return "text"
    if field in PORT_FIELDS or field == "Protocol":
        return "categorical_integer"
    if field in INTEGER_FIELDS:
        return "integer_count"
    # The complete schema is checked before this default can be used.
    return "numeric_measurement"


def field_reason(field):
    """Provide readable reasons for the numerical field-summary table."""
    if field in EXCLUDED_FIELDS:
        return EXCLUDED_FIELDS[field]
    if field == "Label":
        return "Provided annotation supplies the target and is never a predictor."
    if field == "Protocol":
        return "Nominal transport-protocol code; numerical magnitude does not imply order."
    if field in INTEGER_FIELDS:
        return "Traffic count retained numerically; discrete components for later MI analysis."
    return "Traffic measurement retained numerically; continuous components for later MI analysis."


# Nominal codes must not separate otherwise identical traffic measurements.
# Labels and source identifiers are never part of the grouping key.
GROUPING_FIELDS = tuple(
    field for field in PREDICTOR_FIELDS
    if field_kind(field) in {"integer_count", "numeric_measurement"}
)

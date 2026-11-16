# Explainability-guided feature reduction for substation intrusion detection

This study compares full-input and reduced-input multiclass classifiers for the
SANDI-2024 IEC 104 and IEC 61850 datasets, modelled separately. The
comparison uses common SHAP subsets and a separate mutual-information benchmark
at 25% and 50% of active original inputs. The main comparison excludes separate
classifier-specific SHAP reductions and permutation importance.

Each protocol has 26 completed primary test conditions. Matched-observation MI
adds 40 reduced fits and ten test conditions per protocol, giving 36 unique final
conditions in each comparison. Preparation, grouped allocation, training-only
preprocessing, full baselines and primary SHAP/MI reductions are reused. The panel
is Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM, with a
majority-class reference. Validation/test predictions and class/source comparisons
are retained. Each protocol has a ten-round interleaved prediction benchmark and
conditional measured-cost accounting.
All 60 full-versus-reduced pairs across both protocols, including ten sampling-sensitivity pairs per protocol, are consolidated in
`results/comparison/reduction_summary.csv`. No additional training is required
merely to continue analysis and writing.

The results quantify model-dependent performance and cost trade-offs under the
declared representations and selected full-model settings. High normal-fault
false alarms, rare PTP support, selector overhead and dependence within captures
remain material limitations. Small observed differences do not establish formal
performance preservation or deployment reliability. Commands, output definitions,
scientific choices and interpretation limits are documented below.

## Reproducing the reported comparison

Follow the protocol commands below to regenerate the local data and fitted
artefacts from the supplied source CSVs. Fixed allocation, class order, selection
and fitting recipes are part of the comparison. The supplementary matched-MI,
fold-ranking consistency, representation-error and interleaved timing analyses
were introduced after the primary test results had been examined. They provide
exploratory evidence on the same populations, rather than prospective confirmation.

The existing files connect the procedures to their numerical evidence:

| Evidence | Location within each protocol's output folders |
|---|---|
| Allocation and class/group support | `data/<protocol>/assignments.parquet`; `results/<protocol>/partitioning/` |
| Fitted field and component definitions | `results/<protocol>/preprocessing/fields.csv` and `columns.csv`; packet component screens are saved alongside them |
| Chosen model settings and complete fitting recipes | `results/<protocol>/validation/selected_parameters.json` and `reduced_training.json` |
| Per-model ranks, common ranks and selected field identities | `results/<protocol>/feature_selection/`; matched MI uses `sensitivity/feature_selection/` |
| Record-linked probabilities and class decisions | `data/<protocol>/test_predictions.parquet` and `sensitivity/test_predictions.parquet` |
| All fold-pair ranking comparisons and per-round timings | `results/<protocol>/sensitivity/rank_agreement.csv` and `benchmark/prediction_times.csv` |
| Measured selection/refit overhead and prediction-cost offsets | `results/<protocol>/sensitivity/diagnostics/cost_accounting.csv` |

Packet source order and exclusion reasons are in `config/iec61850_sources.csv`
and `config/iec61850_fields.csv`. Selected block identities are recorded in
`results/iec61850/preparation/sampling_blocks.csv`; the numeric grouping flags in
`results/iec61850/preparation/fields.csv` include all ten history keys. Source
ordering and NumPy sampling are defined in `src/iec61850/parsing.py`. Development
payload-layout evidence is in `results/iec61850/representation/sv_payloads.csv`.

`src/checks.py` compares scientific recipes, field identities, actual fitted
preprocessing and estimator parameters, with recorded package-version checks in
the relevant loaders. Supplied replacement commands mark dependent stages stale.
These checks do not cryptographically identify every file or protect against all
unrecorded external edits. The SHAP source correction and reconstruction checks
are described below; they validate numerical consistency, not causal explanations.

SANDI-2024 is attributed to Gutiérrez Mlot et al. (2024),
[Data in Brief](https://doi.org/10.1016/j.dib.2024.111153), whose dataset deposit is
[Zenodo](https://doi.org/10.5281/zenodo.13898982). The article states that the data
files are released under CC BY-NC-SA 4.0. This attribution and the dataset terms
remain applicable to shared material containing dataset content; they are
separate from code and third-party software licences. The local download date
and exact downloaded release identity were not recorded. No current deposit
version is asserted to be the exact local release.

The proposed code upload contains code, configurations, aggregate results and
PNG figures. Raw and prepared observations, models, SHAP arrays and record-linked
`*errors.csv` and `duplicates.csv` files remain local under `.gitignore`; they are regenerated by the
supplied commands. Source-level aggregate counts retain the published recording
names. These exclusions define the upload set and do not establish a legal or
privacy guarantee for every derived artefact.

## Source layout

Protocol-specific scripts live in their own packages. Shared settings and code
stay directly in `src/`; all commands below use these module paths.

```text
src/
├── settings.py                    # shared paths, allocation and model choices
├── checks.py                      # shared safeguard checks
├── partitioning.py                # fixed grouped allocation and support tables
├── modelling.py                   # classifiers, probabilities and pipelines
├── train_full_models.py           # shared command: --protocol
├── train_reduced_models.py        # shared matched reduced-fit runner
├── evaluate_validation.py         # shared command: --protocol
├── evaluate_test.py               # saved final pipelines; protocol-aware uncertainty
├── compare_results.py             # consolidated within-protocol test trade-offs
├── plot_results.py                # shared PNG-only command: --protocol
├── feature_selection.py           # shared sampling, SHAP and rank aggregation
├── shap_backend.py                # shared numerical backend
├── iec104/
│   ├── settings.py
│   ├── prepare.py
│   ├── partition.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train_reduced_models.py
│   ├── evaluate_test.py
│   └── plotting.py                # compatibility entry points for shared plots
├── iec61850/
│   ├── settings.py
│   ├── parsing.py                 # packet parsing, scanning and causal history
│   ├── prepare.py
│   ├── partition.py
│   ├── preprocessing.py
│   ├── sv_payload.py              # development-only payload diagnostic
│   ├── feature_selection.py       # packet SHAP/MI rankings and original-field subsets
│   ├── train_reduced_models.py    # packet reduced-fit entry point
│   └── evaluate_test.py           # packet final-test entry point
└── vendor/shap_0_53_0/             # licensed shared SHAP source
```

Each package also contains `__init__.py`. Prepared data, numerical results and
model files use protocol-specific folders. Saved models refer to the
preprocessor classes in their protocol packages. Shared metric helpers require an explicit class order,
so packet and flow classes cannot be silently interchanged.

Figures are PNG only. To regenerate them from the completed numerical results:

```bash
.venv/bin/python -m src.plot_results --protocol iec104
.venv/bin/python -m src.plot_results --protocol iec61850
```

These commands draw figures; they do not fit models or score test data.

## Software

The existing project environment records Python 3.13.7. Required Python package
versions are listed in `requirements.txt`. Use the existing `.venv` for the
command below; reinstalling the environment is not part of preparation.

For a fresh checkout, create a Python 3.13 environment and install the listed
requirements. Check native-library compatibility before reproducing training
and SHAP analysis on another machine. The preparation command imports only
NumPy, pandas and PyArrow alongside the Python standard library.

## Prepare IEC 104

Run from the repository root, the folder containing this README:

```bash
.venv/bin/python -m src.iec104.prepare
```

`src/iec104/settings.py` contains the source list, label map, complete schema and field
rules. `src/iec104/prepare.py` reads all eight declared sources and checks their
headers, record widths, labels and numeric values. Unexpected tokens stop
preparation. Declared missing tokens and infinities become missing values;
infinities are counted separately. Zero is retained as a value, including in port
and protocol fields.

Negative values in `Flow IAT Min` are treated as unavailable elapsed-interval
measurements and converted to missing. The observation and its other values are
retained. This field-specific rule is declared in `NEGATIVE_TO_MISSING_FIELDS`;
it does not replace negative values in unrelated fields. The affected cell count
is saved in the summary and field table. Filling missing values is deferred to
training-only preprocessing.

Exact duplicates are compared using every parsed original non-target field and
both the original and cleaned label. The first copy in sorted source/record order
is retained. Every row of an original-field vector with conflicting labels is
preserved. This equality check is distinct from the dependency grouping required
for later partitioning.

Flow ID, source/destination IP, source/destination port and absolute Timestamp are
excluded from predictors and omitted from the prepared table. This defines a
port-independent representation; useful service context may be lost and other
recording-specific patterns may remain. Protocol is retained as a nominal code.
Constant-field statistics are descriptive: no field is removed using whole-data
variability. Imputation, scaling, encoding and feature selection must be fitted
within the permitted training scope later.

Each prepared row retains `source_file` and `source_row` so it can be located in
its original CSV. `source_row` is the one-based CSV data-record number after the
header, rather than a physical text-line number. `record_id` combines that relative
filename and record number. `dataset_row` is its zero-based position in the parsed
source collection. `raw_label` preserves the supplied annotation and `class_label`
is the cleaned target. These metadata columns are never model inputs.

Preparation produces:

- `data/iec104/observations.parquet`: eligible inputs, labels and source references.
- `results/iec104/preparation/preparation.json`: actual counts, preparation rules and software versions used.
- `results/iec104/preparation/sources.csv`: source/class counts before and after duplicate handling.
- `results/iec104/preparation/classes.csv`: aggregate class counts and removed duplicate copies.
- `results/iec104/preparation/fields.csv`: field roles, types, reasons and descriptive quality statistics.
- `results/iec104/preparation/duplicates.csv`: removed-record to retained-record links, with headers even when empty.

Numerical results are organised by protocol and operation. The preparation
subfolder keeps these tables separate from subsequent modelling results while
retaining one consolidated table for each purpose.

All outputs are derived locally. Original files in `Datasets/` are opened for
reading only. Existing preparation outputs are protected against accidental
replacement. A deliberate rerun can use `--overwrite`, which replaces only the
six files listed above. Inspect the numerical outputs before partitioning.

## Group and partition IEC 104

```bash
.venv/bin/python -m src.iec104.partition
```

`src/iec104/partition.py` groups all 76 eligible numeric traffic inputs after a
temporary conversion to float32, before any fitted preprocessing or selection.
`GROUPING_DTYPE` declares that precision. Matching missing-value markers are
part of equality. The cast affects only the grouping key: the saved observations
and the later float64 preprocessing inputs are unchanged. Overflow stops grouping.
Protocol is excluded from the grouping key so that a nominal code cannot separate
otherwise identical measurements. Labels and source identifiers never define
the groups. Group identifiers are assignment metadata and are never predictors.

This rule keeps inputs together when small original differences disappear at
the declared numeric precision. It was set before classifier fitting to address
two MITM records with this relationship. The current allocation has 204,612
groups, 255,974 development observations and 63,994 test observations. Every
class is present in each fitting and evaluation population, with no group
crossing. Changing the grouping required replacing assignments and refitting
preprocessing; the seeds were retained without searching alternatives.

The allocation uses `StratifiedGroupKFold` to approximate class balance while
keeping whole groups together. Fold 0 of five shuffled outer folds is reserved
for testing, using seed 42. Three shuffled grouped validation folds are assigned
inside development only, using seed 43. A development row's `validation_fold`
identifies the round in which it is evaluated; it is training data in the other
two rounds. Test rows use `validation_fold = -1` and never enter those rounds.

The target is approximately 80/20 development/test, with a tolerance of five
percentage points in the test fraction. Whole groups prevent guaranteeing exact
per-class proportions. All classes must appear in every fitting and evaluation
population. Infeasible allocation stops the script; it does not search seeds or
folds for a more attractive allocation. Seeds serve reproducibility, not evidence
of independent evaluation.

This is a within-collection evaluation. Equal measurements stay together, but
groups do not establish independence between sessions or captures. Capture-heldout
multiclass evaluation would lose training or evaluation coverage for attack
classes supplied in only one capture. No group is split to force class support.

Partitioning produces:

- `data/iec104/assignments.parquet`: one row per observation, with source references, group, partition and validation fold.
- `results/iec104/partitioning/partitioning.json`: grouping fields, fixed allocation settings, actual counts and separation checks.
- `results/iec104/partitioning/support.csv`: class-level observation/group counts for the collection, development, test and each validation round's training/evaluation populations.

Later operations must reuse these saved assignments and fit transformations and
feature selection on each permitted training population. They must check the
effect of encoding and model precision on matching representations; separation
at the declared precision does not establish separation after every transformation.
This full-input allocation must also be reused for reduced-model comparisons;
do not regroup or delete different evaluation rows for different subsets.

## Fit IEC 104 preprocessing

This operation learns how to prepare model inputs; it does not train a classifier
or produce attack-detection scores. Run it once from the project root:

```bash
.venv/bin/python -m src.iec104.preprocessing
```

The operation completed on the prepared IEC 104 data. All four fitted scopes
retain 72 original inputs and produce 74 encoded columns. The same five
training-constant inputs were removed in each scope. Protocol contributes three
columns, for codes 0, 17 and 6; there were no unseen validation categories. The
bundle and all three numerical tables below have been saved and checked.

`src/iec104/preprocessing.py` first verifies that the prepared observations and saved
assignments still agree, including their current numeric equality groups. It
then fits separate preprocessors for each of the three fold-training populations
and one for all development data. A fold's validation rows do not learn any of
that fold's medians, categories, constant screening or scales. The all-development
preprocessor is for later final fitting and must not be used for fold validation.
Test measurements are read only for checking existing preparation/allocation;
they are excluded from all fitted statistics and transformed-input diagnostics.

The declared rules are:

- Replace missing numeric values with the median of the current training rows.
  Remove fields entirely missing or constant after this training-only filling.
- Add no missing-value indicator columns. This keeps the isolated corrected
  interval from introducing a special marker; potentially useful missingness
  information is therefore not modelled.
- One-hot encode every training-observed Protocol category, including code zero,
  without pooling categories or dropping a reference category. Unseen categories
  produce all-zero Protocol columns and are counted in the diagnostic table.
- Provide unscaled numeric inputs for tree classifiers and a standardised numeric
  view for Logistic Regression. Standardisation subtracts the training mean and
  divides by the training population standard deviation after median filling.
  Binary category columns remain unscaled. Both views use the same fitted state
  and original-field mapping. No clipping or outlier deletion is performed.
- Produce float64 matrices. Count exact validation vectors also present in training
  for both views, plus a float32-cast diagnostic. These counts do not trigger row
  deletion or a new split. A float32 cast does not reproduce every backend's
  internal representation. The matrix limit is for one matrix, not total RAM.

The operation created four files:

- `models/iec104/preprocessors.joblib`: one bundle holding the three fold-training
  preprocessors, the all-development preprocessor, declared panel, rules and
  software versions. Later training code can reuse the appropriate fitted state.
- `results/iec104/preprocessing/fields.csv`: one row per candidate original field
  and training scope, showing retained/dropped status, reason, training missing
  count, learned median/mean/scale or training Protocol categories.
- `results/iec104/preprocessing/columns.csv`: maps each encoded column to its
  original field and records its discrete/continuous role for later analyses.
  Standardising discrete counts for a classifier does not make them continuous
  inputs for MI. Selection uses the declared count roles and treats Protocol as
  one nominal original variable rather than averaging its dummy-column MI.
- `results/iec104/preprocessing/scopes.csv`: training/validation sizes, active field
  and encoded-column counts, unseen Protocol counts and representation-overlap
  diagnostics for both numeric views. The development rows have no evaluation
  population, so their overlap entries are blank.

The current full-input diagnostics show no matching training/validation vectors
at float64 or float32 precision in either numeric view. MITM source records 19
and 23 share a group in development validation fold 2. Both are retained:
they cannot be on opposite sides of a validation round. Allocation uses fixed
seeds 42 and 43.
Float32 grouping does not establish capture independence or reproduce every
backend's processing, and reduction may create additional matching vectors.

No encoded datasets are saved: later code can transform the needed rows using the
small fitted states. Existing outputs are protected; `--overwrite` is only for a
deliberate replacement of these four files. A changed preparation or preprocessing
rule requires reviewing and rebuilding dependent fitted states before training.
The completed full-model fitting stage uses these saved training-fitted states.

The panel provides a linear model, a single tree, bagged trees and two boosting
implementations. Four members are tree-based, so common rankings describe this
panel rather than universal model agreement. The preprocessing command does not
import XGBoost or LightGBM and does not fit classifiers.

## Train full-input IEC 104 models

The full-model training run is complete: 69 logical fits,
69 fitting attempts, 1,056 class-metric rows and 24 selected pipelines. No
convergence retries or fit warnings were recorded in the successful run. The full-training command made no
test predictions. The separate evaluation stage below now contains test results.
The command below is for reproduction; the current
workspace does not need another training run to continue:

```bash
.venv/bin/python -u -m src.train_full_models
```

`src/train_full_models.py` runs the comparison. `src/modelling.py` supplies shared
estimator construction, fitting, aligned probabilities, metrics and saved
pipelines. The exact ordered candidate grids and fixed parameters are declared
in `src/settings.py` and explained in the methodology notes.

Existing fold preprocessors are checked directly against their current training
rows and reused. They are not refitted or overwritten. Test rows remain outside
classifier fitting, tuning, predictions and scores. A selected recipe is fitted
on all development data with the all-development preprocessor only after its
three-fold comparison has finished.

Each learned classifier receives one set of balanced observation weights,
calculated only from its current training labels. The reference remains
unweighted. Metrics use the observed training/validation populations, without
those sample weights. Candidate selection uses mean validation macro F1 across
the three folds; a difference within 0.000001 of the maximum is tied, resolved
by declared candidate order. Training/validation accuracy, weighted F1, normal
false-alarm rate, errors and class-level metrics are also recorded. Validation
scores are selection evidence, not independent estimates of the tuning process.

The estimator seed is 44. Fits run sequentially with at most two CPU workers.
There is no validation-based early stopping. Logistic Regression starts at a
5,000-iteration ceiling; only a fit reporting a convergence warning is retried
at 10,000 then 20,000. Converged fits are retained. Unresolved convergence or
invalid probabilities stop the comparison. The workload is 63 fold fits plus
six all-development fits: 69 logical fits, excluding any convergence retries.

The command creates:

- `results/iec104/validation/full_models.csv`: one row per logical fit, containing
  its model/candidate/scope, parameters, dimensions, fit time, attempts/warnings
  and prefixed training/validation metrics. All-development rows have no
  validation scores. Input-transformation time is shared across candidates in
  one model/scope and must not be summed repeatedly. Prediction time measures
  the estimator on encoded inputs, excluding preprocessing and flow extraction.
- `results/iec104/validation/full_class_metrics.csv`: precision, recall, F1,
  support and errors for every fit's training and validation classes.
- `results/iec104/validation/selected_parameters.json`: declared recipe, software
  versions, candidate mean scores, selected recipes and pipeline locations. The
  status distinguishes running, failed and complete comparisons. Later code must
  require complete status and use its listed paths.
- `models/iec104/full/<model>/fold_0.joblib`, `fold_1.joblib`, `fold_2.joblib` and
  `development.joblib`: 24 selected pipelines across the five classifiers and
  reference, each containing the estimator and its exact fitted preprocessing.
  Selected fold models are retained for SHAP; unselected candidates exist only
  in a temporary folder removed when the command exits.

Saved pipelines are reloaded and checked on development examples. Predicted
classes must match exactly; probabilities must agree within absolute/relative
tolerance 0.000000000001. This permits rounding differences from parallel tree
summation without accepting a material prediction change.

The CSVs and settings record are updated during training. An incomplete run is
not a final result, and automatic resumption is not implemented. A deliberate
rerun uses `--overwrite`, replacing only known full-training outputs; base data,
assignments and preprocessors are preserved. An incomplete overwrite may leave
older pipeline files, which must not be used unless the current settings record
lists them and reports complete status. Feature selection requires a complete
full-training record and follows the ranking and reduction rules below. Detection changes are reported directly without an arbitrary
retention pass/fail margin or a claim of statistical equivalence.

Selected mean validation macro F1 is 0.929119 for Logistic Regression, 0.980622
for Decision Tree, 0.971343 for Random Forest, 0.976496 for XGBoost, 0.981134 for
LightGBM and 0.121930 for the majority reference. These are development-selection
results, not held-test estimates. The methodology notes include selected settings
and rare-class limitations. Saved full models can be reused for explanations and
later evaluation; reduced representations need their own fits.

## Research question and objectives

To what extent can common SHAP-selected feature subsets reduce input requirements
and computational cost while preserving multiclass detection performance,
compared with full-input models and mutual-information-selected subsets, on the
SANDI-2024 IEC 104 and IEC 61850 datasets?

The supporting questions concern (1) which original fields SHAP and MI select,
(2) aggregate and class-specific detection changes relative to full inputs, and
(3) measured computational savings and consistency across the panel and protocols.

The objectives are to establish full-input baselines; identify influential
original fields using training-only SHAP and a separate MI benchmark; fit
reduced-input models under matched conditions; and evaluate class-specific
detection and measured computation. Each protocol has its own common subsets.
The contribution is a controlled application and evaluation of established
methods, not a newly invented SHAP algorithm or a claim of universal superiority.

## Recover full-model validation predictions

```bash
.venv/bin/python -u -m src.evaluate_validation
```

This loads the saved selected fold models without fitting. Every development row
is predicted once by its held-fold model. Recovered metrics must reproduce the
full-training tables. Pooled held-fold metrics and mean fold metrics are distinct.

- `data/iec104/full_validation_predictions.parquet`: record/group identities,
  true/predicted classes and aligned probabilities for later paired comparisons.
- `results/iec104/validation/full_selected_metrics.csv` and
  `full_selected_class_metrics.csv`: selected fold/pooled metrics, class support
  and repeated full-pipeline prediction times, separate from tuning candidates.
- `full_confusion_matrices.csv` and `full_errors.csv` in the same folder:
  confusion counts and record-linked error details.
- `prediction_recovery.json`: completion and verification counts.

The recovery completed: 1,535,844 predictions across six model families and
255,974 development records per family. All selected fold metrics reproduced.
LightGBM's first fold has five non-MITM records predicted as MITM, one each from
DoS, flood, starvation, NTP DDoS and port scanning. Its pooled MITM precision is
0.80 and recall is 20/21. These observations motivate reporting precision and
class errors; they do not trigger a label change or a new split.

## Select training-only features

```bash
.venv/bin/python -u -m src.iec104.feature_selection
```

The three fold-training populations and all-development population are analysed
separately. All classifiers in one scope share the same deterministic samples.
Foreground samples contain up to 128 real training rows per true class;
background samples contain up to 16 per class. Classes below a cap retain all
available rows without duplication. The seed is 45 with separate scope/role
streams. Sampling bounds explanation cost; it is not a proven optimal sample size.

SHAP uses explicit independent-component backgrounds. Trees use interventional
TreeExplainer; Logistic Regression uses LinearExplainer. Explanations reconstruct
class probabilities for DT/RF, decision scores for LR and margins for boosting.
A local MIT-licensed SHAP 0.53.0 threshold-precision correction is built by
`src/shap_backend.py` from `src/vendor/shap_0_53_0/`; a C++ compiler is required.
It follows https://github.com/shap/shap/pull/5122 and preserves float64 split
thresholds. The installed package is unchanged. An independent check against
actual model outputs uses absolute/relative tolerance 0.00001. Contributions of encoded components are summed with their
signs within each original field before taking absolute values. Importance is
averaged equally across output classes, then within each true class, then equally
across true classes. This single class-balanced summary is aligned with macro F1;
it is not a separate balancing experiment and does not create more rare evidence.
Aggregated component SHAP is not jointly masked group SHAP; correlated inputs
and background composition can affect the ranking.

Each model's fields receive average-tie ranks. Common SHAP selection uses equal
model mean ranks, with exact doubled-integer rank sums and alphabetical final
ties. It describes this five-member panel, four of which use trees.

MI uses up to 2,000 training rows per true class, sample seed 46, three neighbours
for continuous estimation and estimator seed 47. Numeric fields use their
training-fitted missing-value replacement; semantic counts are discrete and
Protocol is one nominal integer-coded variable. Capping retains residual sample
imbalance without duplicated rare examples. MI is univariate, sensitive to its
sample distribution and potentially biased for high-cardinality discrete counts;
it does not explain a fitted classifier.

The command writes consolidated `rankings.csv`, `common_ranking.csv`,
`importance_by_class.csv`, `selected_features.csv` and `selection.json` under
`results/iec104/feature_selection/`. The JSON records declared rules, actual sample
sizes, timings, reconstruction checks and running/failed/complete status. Sample
identities are in `data/iec104/feature_selection/samples.parquet`; local SHAP
arrays are under its `shap/` subfolder. No hashes or per-job receipt framework are
used. Later code requires complete status and matching settings.

Reduced models retain all encoded components of each selected original field.
The denominator is the training-active field count, rounded upward: IEC 104
currently yields 18 and 36 fields from 72. Reduced models reuse the selected
full-model hyperparameters and fixed assignments; they are not separately tuned.
Current validation scores remain development-selection evidence. No current
test scores may influence selection or fitting.

## Fit and compare reduced models

```bash
.venv/bin/python -u -m src.iec104.train_reduced_models
```

This requires complete selection and recovered full predictions. It fits five
classifiers, two selectors and two feature budgets in three fold-training scopes
and one all-development scope: 80 logical fits. No new hyperparameter grid is
searched. Each reduced preprocessor projects the corresponding full training
state, retaining every component of selected original fields. Projection and
saved-pipeline predictions are checked on real training examples from all classes.

Outputs are consolidated under `results/iec104/validation/`:

- `reduced_models.csv`: fit settings, dimensions, attempts, warnings, timings and
  training/validation scores; `reduced_class_metrics.csv`: class metrics.
- `reduced_selected_metrics.csv` and `reduced_confusion_matrices.csv`: fold and
  pooled held-fold outcomes, with repeated full-pipeline prediction timings.
- `reduction_comparisons.csv` and `class_reduction_comparisons.csv`: paired
  reduced-minus-full metric changes, newly wrong/corrected records and class effects.
- `representation_overlap.csv`: matching reduced training/validation vectors at
  float64/float32 precision, measured once per scope/condition/numeric view.
  These are diagnostics; all declared evaluation observations are retained.
- `reduced_training.json`: recipe, complete/failed status and saved pipeline paths.

Saved reduced pipelines are under
`models/iec104/reduced/<selector>_<percentage>/<model>/<scope>.joblib`.
Record-linked validation probabilities are in
`data/iec104/reduced_validation_predictions.parquet`. No test predictions occur.
Only all-development reduced models are used for final test evaluation;
fold models support verification and development analyses.

Existing operation outputs require a deliberate `--overwrite` to replace them.
A failed status is not a usable completed comparison. These scripts do not
implement automatic job resumption or a separate receipt framework.

## Plot saved results

```bash
.venv/bin/python -m src.plot_results
```

This reads saved numerical tables and writes PNG figures under
`figures/iec104/`. Full-model plots show fold macro F1, pooled class precision and
recall, and confusion counts. The completed selection adds a common-rank plot.
Reduction figures show performance curves, class-recall changes and
descriptive fitting/prediction/model-size ratios. Plotting does not refit or score test data.

## Interpretation and validity

Reserved test observations were excluded from model fitting and feature
selection. Grouped evaluation used the supplied collection and did not establish
independence between captures.
The results describe the supplied recordings; performance on other substations
or attack scenarios could differ. The literature includes the published 2026
Arreche common-ranking article.

## Evaluate the reserved IEC 104 test portion

```bash
.venv/bin/python -u -m src.iec104.evaluate_test
```

This loads all-development pipelines only: full plus SHAP25/50 and MI25/50 for
all five classifiers, and the majority reference (26 conditions). The comparison
plan is written before predictions. There is no fitting, feature selection,
threshold adjustment or removal of an unfavourable condition. “Reserved”
describes the observations' role in the implemented fitting/evaluation procedure.

The output `data/iec104/test_predictions.parquet` retains identities, groups,
probabilities and predictions. Consolidated files in `results/iec104/test/` are
`point_metrics.csv`, `class_metrics.csv`, `confusion_matrices.csv`, `errors.csv`,
`paired_comparisons.csv`, `class_paired_comparisons.csv`, `paired_intervals.csv`,
`class_recall_intervals.csv`, `representation_overlap.csv` and `evaluation.json`.
The last file records plan, completion status, condition coverage and actual
uncertainty support. Overlap diagnostics retain every evaluation observation.

Paired uncertainty uses 1,000 fixed-seed (48) whole-group resamples stratified by
true class. The same group multiplicities apply to all conditions. The sparse
error calculation is checked against every actual point metric before resampling.
At least two groups per class are needed to avoid a completely degenerate
stratum; this is a mathematical guard, not a validated adequacy threshold.
Five MITM groups still provide very weak support for precise inference. Percentile
intervals are descriptive and conditional on these observed measurement groups,
class-group counts, trained models and chosen representation. They do not cover
independent-capture variation, model refitting or feature-selection uncertainty;
matching reduced vectors do not become independent merely through this bootstrap.
The analysis makes no formal equivalence or operational safety claim.

IEC 104 test execution is complete: all 26 conditions, 1,663,844 saved predictions
and 1,000 conditional paired group resamples. Every condition's identities,
metrics and confusion counts were checked from probabilities. Results are mixed;
all declared conditions are reported. Validation improvements are not assumed to
carry through to test results. No test-based retuning or feature-budget change
was performed. Test figures are regenerated by `src.plot_results`.

## Prepare IEC 61850

```bash
.venv/bin/python -u -m src.iec61850.prepare
```

`src/iec61850/settings.py` declares five scenario classes, source cap 25,000,
16-packet blocks, sampling seed 49 and eight previous packets of source-block
context. `src/iec61850/parsing.py` reuses the reviewed bounded scanner, typed parser
and history functions. `config/iec61850_sources.csv` supplies expected source
counts that are freshly checked; `config/iec61850_fields.csv` declares field
roles/types. No archived scores, ranks, fitted objects or encoded data are inputs.

The normal preparation command fully reads all nine CSVs for schema, width,
label and count checks; only the fixed block sample is retained. Explicit cached
reuse avoids repeating that scan when the validated sources and sampling match.
The declared full-input pool uses the authors' 167 candidate fields minus 57
explicit exclusions, plus ten history units. It
does not cover all 1,194 raw dissector columns. Packet repetitions are retained.
Whole history blocks and equal base numeric/history vectors at float32 precision
are connected into dependency groups. Payload detail does not split these groups. Stored measurements retain original precision.
No imputation, category fitting, scaling or selection occurs in preparation.

Outputs are `data/iec61850/observations.parquet` and `features.json`, with
`preparation.json`, `sources.csv`, `fields.csv`, `classes.csv`, `sampling_blocks.csv`
and `components.csv` under `results/iec61850/preparation/`. These provide runtime
inputs, original-field roles, actual sample/support counts and dependency checks.
Small derived source samples live in the ignored cache. After confirming a
process has stopped, `--resume` reuses completed scans only when settings and
source size/mtime agree. `--overwrite` deliberately rebuilds preparation outputs.

The sample changes source/class prevalence. History is short capture-order
context, with state reset at each block; it does not establish long-term replay
or endpoint-specific detection. Supplied filename labels identify scenarios,
not independently verified maliciousness of every packet. Grouped partitioning
must establish feasible class support before fitting, without weakening groups
or dropping the PTP class to force a result.

Preparation is complete and its saved outputs have been checked. The source
scan validated all 22,075,596 raw records across nine sources. The prepared
representation contains 200,095 records, including the declared SV payload input.
There are 120 candidate original fields, 12,506 source
blocks and 10,050 linked dependency groups. The PTP class has 159 records in nine
dependency groups. All saved record identities, numeric-vector groups and ten
history values per observation were checked. Preparation itself fits no classifiers.

## Completed study

Both protocols have completed preparation, grouped allocation, training-only
preprocessing, full-model fitting, SHAP/MI selection, 80 reduced fits and all 26
final test conditions. The consolidated comparison contains 50 full-to-reduced
pairs. Saved results and PNG figures support analysis and writing without
repeating training.

IEC 61850 has 160,095 development and 40,000 test records. All five classes occur
in every training/validation population, but its 32 PTP test records occupy one
dependency group. Point metrics remain available; primary all-class group
uncertainty is not estimable from that support.

All commands below reproduce completed stages. Shared modelling and metric
helpers require explicit protocol-specific class counts or labels. Shared
command-line tools default to IEC 104 unless the packet protocol is supplied;
use the packet entry points or the explicit protocol argument for IEC 61850.

## Allocate IEC 61850 dependency groups

```bash
.venv/bin/python -u -m src.iec61850.partition
```

This reuses the shared fixed grouped-allocation logic with prepared packet
components. No model is fitted. The saved `data/iec61850/assignments.parquet`
defines each record's group, development/test population and validation fold.
`results/iec61850/partitioning/support.csv` reports class rows/groups for every
fitting and evaluation population; `partitioning.json` records the procedure,
actual totals and checks. The same outer/inner seeds 42/43 and fixed outer fold
zero are used without a seed or fold search. Independent regeneration reproduced
every saved assignment, with no source-block or measurement-group crossing.

Development PTP support is 127 packets in eight groups. Validation folds have
48/32/47 PTP packets in 3/2/3 groups; their corresponding training populations
have 79/95/80 packets in 5/6/5 groups. This is within-collection dependency
separation, not independent-capture evaluation.

## Fit IEC 61850 preprocessing

```bash
.venv/bin/python -u -m src.iec61850.preprocessing
```

The typed packet encoder adapts the reusable list/category preprocessing.
Ordered lists have a fixed limit of 64 positions plus length/presence components;
overlong lists stop the operation rather than being truncated. Only training
rows determine medians, numeric scales, category vocabularies and omitted
constant/all-missing components. Varying training missingness is retained.
Numeric value columns are scaled for Logistic Regression; tree/reference inputs
remain unscaled. One-hot and missing-indicator columns are never standardised.

A field vocabulary has at most 128 retained tokens including observed missing
and any pooled bucket. When the limit is exceeded, training-frequency counts
choose retained tokens with lexical ties. Lists share a field vocabulary across
fixed positions. Unseen nonmissing tokens use an existing pooled bucket or an
all-zero encoding. Transformations never learn new categories from validation
or test data. Padding contributes to training token counts, so vocabulary
selection is a representation choice rather than a neutral lossless encoding.

Outputs are `models/iec61850/preprocessors.joblib` (the three fold-training states
and all-development state) and five CSVs in `results/iec61850/preprocessing/`:
`fields.csv` reports retained/omitted original fields; `columns.csv` maps encoded
columns to fields/components; `component_screen.csv` lists omitted components;
`scopes.csv` reports dimensions, timings and exact encoded overlap at float64/32;
`unknown_categories.csv` counts validation tokens absent from training, including
fields omitted as constant. No encoded matrices or test transformations are
saved. The per-matrix budget is 2 GiB; this is a guard, not a memory measurement
for the whole Python process. Saved states are checked against training data
and their transformations are checked after reload on real development examples.

The completed packet preprocessing retains 68 original fields in every scope.
The other 52 are entirely missing in that scope's training rows. Encoded widths
are 387/388/389 for the three folds and 389 for all development data. The payload
contributes 38 encoded columns under the single original field `sv.seqData`. No complete
encoded validation vector matches a training vector in either numeric view at
float64 or float32 precision. This does not establish independence of captures.

The representation treats `goose.Data` as a list of nominal payload
type codes and `ptp.v2.logmessageperiod` as a nominal interval/control code.
Neither belongs to the 47-field numeric grouping key. Saved record identities,
measurement/history values and assignments were checked. The preprocessing
states and their five CSV summaries describe the same fixed allocation used by
all full and reduced conditions.

## Train full-input IEC 61850 models

```bash
.venv/bin/python -u -m src.train_full_models --protocol iec61850
```

This reuses the full-model training loop with the packet input loader and the
explicit five-class order. The candidate grids, balanced training observation
weights, seed 44, two-worker limit, macro-F1 selection and convergence-only LR
retry policy are shared with IEC 104. Reduced conditions use the selected
full parameters, without a separate tuning search. The selected validation
scores guide development and are not independent estimates of the whole search.

The completed workload contains 69 logical fits and 660 class-metric rows, with 24
selected fold/development pipelines. Outputs are `full_models.csv`,
`full_class_metrics.csv` and `selected_parameters.json` under
`results/iec61850/validation/`, plus pipelines under
`models/iec61850/full/<model>/<scope>.joblib`. The JSON must report `complete`
before those objects are used for feature selection. Interrupted/failed records
are not completed experiments. No test scores or SHAP rankings are produced by
training. The IEC 104 command and its existing saved recipe remain compatible.

To replace the interrupted packet run with the corrected representation, use:

```bash
.venv/bin/python -u -m src.train_full_models --protocol iec61850 --overwrite
```

After successful fitting, verify and retain held-fold predictions with:

```bash
.venv/bin/python -u -m src.evaluate_validation --protocol iec61850
```

This reloads the selected fold pipelines without fitting or test scoring.
Recovered metrics must reproduce their training-table validation scores.
`data/iec61850/full_validation_predictions.parquet` retains probabilities,
record identities and groups. `full_selected_metrics.csv`,
`full_selected_class_metrics.csv`, `full_confusion_matrices.csv`, `full_errors.csv`
and `prediction_recovery.json` in `results/iec61850/validation/` contain fold and
pooled summaries, class errors and completion checks. Three repeated predictions
after a short warm-up give descriptive full-pipeline prediction times.

Generate the available packet validation figures from those saved tables with:

```bash
.venv/bin/python -m src.plot_results --protocol iec61850
```

This writes three PNG figures in `figures/iec61850/validation/`: fold macro F1,
pooled class precision/recall and pooled confusion counts. It performs no model
fitting or predictions. The same plotting command also regenerates completed
selection, reduction and test figures from their corresponding numerical results.

The full-input baseline includes SV payload: 69
logical fits in 69 attempts, 660 class-metric rows and 24 selected pipelines.
There were no fit warnings or convergence retries. These are the active
references for the full-versus-reduced comparison.

| Classifier | Mean validation macro F1 | Pooled normal false-alarm rate |
| --- | ---: | ---: |
| Logistic Regression | 0.609628 | 40.17% |
| Decision Tree | 0.636554 | 34.93% |
| Random Forest | 0.776753 | 35.10% |
| XGBoost | 0.796770 | 34.90% |
| LightGBM | 0.769632 | 35.07% |
| Majority reference | 0.109023 | 0.00% |

Macro F1 is the mean across the three validation folds. The false-alarm column
uses pooled held-fold predictions. The majority reference predicts normal for
every record, so its zero normal false-alarm rate comes with no attack detection.

Recovery saved 960,570 predictions, one per development row per family. All
18 selected fold populations reproduced the training metrics. Independent
recalculation checked identities, folds, class order and probabilities, all 24
fold/pooled overall rows, 120 class rows, 600 confusion rows and 277,627
record-linked errors. Three PNG figures were regenerated; the macro-F1 and
class-precision/recall figures were visually inspected.

Normal false alarms remain a serious weakness: 34.90–40.17% across learned
models. Within the normal-fault capture they range from 93.54% to 99.61%, while
the main normal capture has no observed false alarms. These results do not
establish an operationally reliable detector. Source-level errors are saved under
`results/iec61850/representation/source_errors.csv`.

The full baselines, SHAP/MI selection, reduced fitting and test scoring all use
the same saved packet population, group IDs and allocation.

## Inspect development SV payload coverage

```bash
.venv/bin/python -m src.iec61850.sv_payload
```

This development-only diagnostic reuses a strict bounded hexadecimal parser.
It reads cached source strings filtered to the saved development assignments,
and aggregates existing held-fold predictions. It does not refit classifiers,
generate predictions, inspect test payloads or change the active representation.
The source files must retain their scanned size/mtime, and cached identities,
labels, ASDU metadata and validation memberships are checked.

Three small files are saved under `results/iec61850/representation/`:

- `sv_payloads.csv`: payload availability, lossless byte shapes, malformed-input
  counts and ASDU consistency for each development source.
- `source_errors.csv`: existing held-fold errors by classifier and source,
  including normal-source false-alarm rates. Sources are analysis metadata only.
- `representation.json`: parser bounds, actual review counts and scope checks.

Existing outputs require a deliberate `--overwrite`. The parser validates byte
structure; it does not decode physical measurements.

The completed review covers all 160,095 development records. Exactly 82,171 have
payloads, all with two 32-byte ASDU parts; 77,924 have no payload. No malformed
hexadecimal input or ASDU metadata inconsistency was found, and lossless byte
roundtrips passed. The available `sv.meas_value`, `sv.meas_quality`, `sv.datSet`,
`sv.smpRate` and `sv.smpMod` fields are empty. Normal-fault payload availability
is 21,202/21,248 development records. All 54 source-metric rows reproduce the
existing pooled error counts and normal false-alarm rates.

The SV payload representation retains `sv.seqData` as one original input expanded
into ordered unsigned bytes. The development review established a supported
layout of two ASDUs with up to 32 bytes each. Parsing preserves leading zeroes
and ASDU boundaries and rejects larger layouts rather than truncating them.
Preprocessing includes payload presence, ASDU count and lengths, learns byte
medians from training only and screens constant components. Presence and lengths
identify absent byte positions without repeating a missing flag for every byte.
No channel names, engineering units or scaling are inferred. Byte magnitudes are
not physical distances and may carry capture-specific patterns. The representation
preserves available input; it does not establish a detection improvement.

The sampled records, 47 base numeric/history grouping inputs and whole dependency
groups are retained. Payload bytes do not split those conservative components.
The fixed allocation is reproduced and checked against every saved assignment;
there is no search for another split. SHAP/MI grouping treats the whole
payload expansion as one original field.

To reproduce preparation from an unchanged validated cached sample:

```bash
.venv/bin/python -m src.iec61850.prepare --overwrite --reuse-samples
.venv/bin/python -m src.iec61850.partition --overwrite
.venv/bin/python -m src.iec61850.preprocessing --overwrite
```

Cached reuse checks source size/mtime, scan completion, source labels, the seeded
sampling plan and each sampled source-row identity. A fresh checkout without
those caches should use the normal preparation command instead. Derived model
results must match the declared representation before feature selection.



## Select IEC 61850 original fields

```bash
.venv/bin/python -u -m src.iec61850.feature_selection
```

The packet selector reuses shared SHAP sampling, reconstruction and consensus
routines extracted from the completed flow selector. It loads the saved selected
full models; it does not fit classifiers or predict test observations. Each fold
uses its own training-only preprocessing and training rows, then a final analysis
uses all development rows. The model recipes were selected using development
validation, so these explanations are conditional on those recipes.

SHAP uses at most 128 foreground and 16 background rows per true class, with the
declared fixed seeds and no repeated rare records. Within each original field,
signed encoded contributions are summed before absolute values. Magnitudes are
averaged across output classes, within each true class, and then equally across
true classes. Equal model ranks form the common ranking. This aggregates encoded
component SHAP values; it is not jointly masked coalition-group SHAP. The
reconstruction check must pass before an explanation is accepted.

MI uses at most 2,000 training rows per true class, retaining residual imbalance.
The saved medians, pooling and active-component rules are reused. Nominal
components are scored jointly as categories rather than separate one-hot bits;
counts, bytes and missingness indicators are discrete. The field score is the
maximum component MI. This retains a field if one component is informative,
does not sum redundant components and does not dilute one informative position
among many uninformative positions. An average is a reasonable alternative with
that dilution limitation. A maximum gives wide fields more opportunities for a
high score; this is not joint multivariate MI or an unbiased comparison of field
widths. Component scores/counts are saved. SHAP and MI are two configured
procedures with different sampling/aggregation, not an isolated causal test of
one explanation method against another.

The current 68 active fields give 17- and 34-field subsets at 25% and 50%.
Selecting `sv.seqData` retains all 38 fitted payload columns as one original
field. The same rule keeps every fitted encoded column of other selected fields.
Group assignments and full-model parameters are not changed by selection.

Saved outputs:

- `results/iec61850/feature_selection/rankings.csv`: five model-specific SHAP
  rankings and the MI field ranking in each scope.
- `common_ranking.csv`: equal-model common SHAP ranks and deterministic order.
- `importance_by_class.csv`: SHAP field importance for every true class/model.
- `selected_features.csv`: the original-field subsets used by later reductions.
- `mi_components.csv`: logical component types, sample diversity and MI scores.
- `selection.json`: declared settings, actual scope progress, timings and checks.
- `data/iec61850/feature_selection/samples.parquet`: original row/group identities
  for each scope and sampling role.
- `data/iec61850/feature_selection/shap/<model>/<scope>.npz`: encoded/grouped
  contributions, base values and reconstructed model outputs for later checks.

Existing outputs require a deliberate `--overwrite`. To draw the common ranking
from saved numerical results, use the shared PNG plotting command:

```bash
.venv/bin/python -m src.plot_results --protocol iec61850
```

Reduced-model fitting and test evaluation use the separate reproduction commands below.


The packet selection run is complete: 20 SHAP analyses and four MI estimates,
with no recorded warnings. Independent checks reproduced sampling identities and
training boundaries, signed original-field grouping, class averaging, model
ranks, common ordering, component-MI maxima and every subset. Results contain
1,632 ranking rows, 272 common-rank rows, 6,800 class-importance rows, 408 selected-
field rows and 1,032 MI-component rows. The 35,130 sample-role records represent
permitted training selections; rows may recur across roles/scopes, with no
replacement inside a role/class sample. Maximum SHAP reconstruction error was
5.70e-6, within the declared tolerances. No classifier was refitted and no test
records were explained or scored.

Final all-development input sets, against the full 68 fields / 389 columns:

| Selector | Original fields | Encoded columns |
| --- | ---: | ---: |
| SHAP | 17 | 230 |
| SHAP | 34 | 315 |
| MI | 17 | 233 |
| MI | 34 | 265 |

`sv.seqData` ranks fifth in the final common SHAP ordering and is included in all
four final subsets. The first four common fields are preceding-packet frame-
length mean, standard deviation, maximum and minimum. These are model-attribution
findings in the declared representation, not physical causal discoveries.

Across the three folds, SHAP17 shares 15 fields and SHAP34 shares 32. Final SHAP
and MI sets overlap in 13 of 17 and 23 of 34 fields. These limited-fold summaries
do not establish universal stability. SHAP17 widths vary from 97 to 233 across
folds because `goose.Data` is outside the first-fold budget but inside the other
two; that field alone expands into 137 fitted columns. Original-field count
must not be treated as encoded width or a direct cost reduction.

Recorded all-scope analysis time was 2,406.99 seconds for SHAP and 3.64 seconds
for MI. The final development-scope values were 666.81 and 0.88 seconds. SHAP's
timer includes transformation, explanation and compressed-array saving; MI's
includes logical-component construction and scoring, without a comparable array
save. Timers exclude prior classifier fitting and are descriptive local wall
times. Selector overhead must accompany any later claim about computational
savings; this is not evidence of lower total study cost, energy or deployment
cost. The ranking PNG is `figures/iec61850/feature_selection/common_shap_ranking.png`.

The frozen subsets were fitted with the selected full-model recipes,
as described below. The full baseline's high normal-fault error remains a
limitation of both full and reduced classifiers.


## Fit IEC 61850 reduced models

```bash
.venv/bin/python -u -m src.iec61850.train_reduced_models
```

This uses the shared loop adapted from the completed IEC 104 implementation.
The saved 17/34-field SHAP and MI sets are frozen per training scope. No separate
reduced hyperparameter search is performed. Each preprocessor is projected from
its own training-fitted full state and checked against the matching full encoded
columns; packet rules/vocabularies are filtered with the selected fields. The
same observations, group assignments, model seeds, class weights and numeric
views are used. No preparation, SHAP estimation or full-model fitting is repeated.

The run fits 80 pipelines: five classifiers, two selectors, two budgets and four
training scopes. It saves their held-fold probabilities, multiclass and class
metrics, paired full/reduced differences and source-level errors. The normal-
fault capture remains a required source-specific review. All saved development
rows are retained in comparisons. Encoded overlap is a diagnostic and does not
trigger deletion or a new allocation.

Outputs are 80 `models/iec61850/reduced/<condition>/<model>/<scope>.joblib`
pipelines, `data/iec61850/reduced_validation_predictions.parquet`, and consolidated
files under `results/iec61850/validation/`:

- `reduced_models.csv`, `reduced_class_metrics.csv`: fit-time training/validation
  metrics, costs, attempts and warnings. Fit rows include saved/full pipeline
  byte counts and full encoded widths.
- `reduced_selected_metrics.csv`, `reduced_confusion_matrices.csv`: saved-pipeline
  fold/pooled validation measurements and confusion counts.
- `reduction_comparisons.csv`, `class_reduction_comparisons.csv`: paired overall
  and per-class changes relative to the same full classifier.
- `representation_overlap.csv`: matching reduced training/validation vectors;
  both numeric views and precisions, with zero rows removed.
- `source_reduction_comparisons.csv`: paired source-level errors and normal
  false-alarm rates, including the normal-fault capture.
- `reduced_training.json`: execution status, fixed recipes, indexed pipelines,
  completed fit counts and any failed-fit details.

Existing reduced outputs require deliberate `--overwrite`. The shared plotting
command reproduces PNG reduction performance, class recall, recorded costs and
normal-source error figures from the completed results. Test scoring uses the
separate command below and does not repeat training.

The run completed with 80 fits, 80 attempts, 80 saved pipelines and no fit
warnings or convergence retries. All 3,201,900 saved validation predictions,
aggregate/class/confusion metrics and paired overall/source comparisons were
independently checked. The following values are means across three validation
folds, not test results:

| Classifier | Full | SHAP 17 fields | SHAP 34 fields | MI 17 fields | MI 34 fields |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.609628 | 0.579412 | 0.607416 | 0.582695 | 0.577628 |
| Decision Tree | 0.636554 | 0.682216 | 0.636673 | 0.715587 | 0.719843 |
| Random Forest | 0.776753 | 0.739408 | 0.775917 | 0.764864 | 0.770292 |
| XGBoost | 0.796770 | 0.782102 | 0.797558 | 0.784404 | 0.789839 |
| LightGBM | 0.769632 | 0.750207 | 0.767791 | 0.751107 | 0.764750 |

SHAP34's all-development classifier fitting times were 48–57% lower and its
summed fold median pipeline prediction times were 52–59% lower than the saved
full references. These are descriptive local timings with different execution
times, not repeated controlled benchmarks. Tree pipeline sizes changed little;
some other reduced conditions were slightly larger. One-off selector costs
reported above are additional. No total deployment saving is established.

SHAP does not dominate MI: MI is substantially better for Decision Tree here.
SHAP17 also reduces Random Forest PTP recall from 0.6457 to 0.5197 in pooled
validation, despite only 16 additional errors overall. The reduced normal-fault
false-alarm rates remain about 93.4–99.7%. Four PNGs in
`figures/iec61850/reduction/` show macro F1, class recall changes, computation
ratios and normal-source false alarms. All conditions remain in the comparison;
there is no post-result budget or subset adjustment. The 26 saved final packet
conditions were evaluated as described below. Unsupported all-class group
uncertainty is explicitly suppressed; the flow bootstrap is not applied unchanged.


## Evaluate final IEC 61850 models

This stage is complete. The command below is for reproduction; current outputs
must not be rerun simply to obtain different results:

```bash
.venv/bin/python -u -m src.iec61850.evaluate_test
```

The shared `src/evaluate_test.py` adapts the completed flow evaluator. Protocol
entry points retain the existing flow command. It loads saved final models and
checks their class order, selected recipes and projected preprocessing state;
it does not fit, tune, select features, calibrate or change the allocation.
Existing test outputs require deliberate `--overwrite` for reproduction.

Every condition predicts the same 40,000 test records in 1,965 dependency groups.
All 26 conditions completed, and all 1,040,000 saved predictions were checked
against original identities, probability validity, exact error/confusion counts,
aggregate/class scores and paired class/source differences. No classifier fit
was repeated. The outputs are:

- `data/iec61850/test_predictions.parquet`: all test probabilities and identities.
- `results/iec61850/test/point_metrics.csv`: 26 model/condition metrics and costs.
- `class_metrics.csv`, `confusion_matrices.csv`, `errors.csv`: class measurements,
  exact confusion counts and record-linked mistakes for every condition.
- `paired_comparisons.csv`, `class_paired_comparisons.csv`: 20 overall and 100
  class comparisons against the matching full model.
- `source_metrics.csv`, `source_paired_comparisons.csv`: 234 source measurements
  and 180 paired source comparisons, including all three normal sources.
- `support.csv`: test record/group counts for each class. Class group counts
  overlap where a dependency component contains multiple labels.
- `representation_overlap.csv`: ten condition/view diagnostics; zero rows removed.
- `paired_intervals.csv`, `class_recall_intervals.csv`: headers only because
  primary all-class uncertainty is unsupported, not missing execution output.
- `evaluation.json`: frozen evaluation plan, indexed conditions, completed status
  and the explicit uncertainty exception.

All 32 PTP test packets belong to one group; 206 test groups contain mixed labels.
The primary all-class group interval is marked `not_estimable`, with 1,000
requested and zero executed resamples. There is no individual-row bootstrap,
class dropping or artificial split of mixed groups. Point results remain
available for all five classes. This does not imply equivalence or strong
independent-capture evidence.

SHAP34's test macro F1 is within 0.005 of each full reference. MI17/34 is better
for DT by about 0.113, while selectors/budgets behave differently for other
models. Class-level changes remain relevant; LR fuzzy recall falls by about
2.21 percentage points under SHAP34. Normal-fault false alarms remain about
94.6–99.8% among learned conditions. This limits practical reliability claims.
Observed SHAP34 test prediction savings are about 11–23%; earlier validation
ratios differ and both are descriptive local timings. Selector overhead remains
additional, and tree pipeline sizes change little.

Regenerate packet figures from saved tables:

```bash
.venv/bin/python -m src.plot_results --protocol iec61850
```

Six PNGs in `figures/iec61850/test/` show macro F1, paired point differences,
class precision/recall changes, cost ratios, full confusions and normal-source
false alarms. The paired figure explicitly states that intervals are unsupported.
All figures are PNG files generated from the numerical results.

## Consolidate the two protocol comparisons

```bash
.venv/bin/python -m src.compare_results
```

This reads completed numerical outputs only. It creates/replaces
`results/comparison/reduction_summary.csv` (60 paired conditions, with main and sampling-sensitivity roles identified) and
`figures/comparison/performance_prediction_tradeoffs.png`. It does not load raw
data, make predictions or fit anything. Field/encoded widths, performance and
class changes, normal false alarms, cost ratios and uncertainty status remain
separate by protocol. Shared selector seconds repeat for context and must not
be summed across rows; the SHAP and MI timers have different boundaries.
No cross-protocol pooled score or universal best condition is calculated.

Both protocols supply completed evidence for all four objectives. Remaining work
concerns thesis synthesis and artefact/citation review. The methodology
documentation describes the procedures and calculations separately from numerical
findings and their interpretation.


## IEC 104 sampling sensitivity and repeated prediction

The main MI benchmark uses a larger capped training sample than SHAP. An additional
`mi_matched` condition estimates MI on exactly the saved SHAP foreground identities,
so this comparison can examine the effect of selector sampling without rebuilding
the observations, allocation, preprocessors or full models. It retains the MI
estimator, variable types, selected full-model parameters and 18/36-field budgets.
Matching observations does not make empirical MI class proportions equal to SHAP's
equal-class attribution average. Both configured MI conditions remain in the study.

Run these operations from the project root:

```bash
.venv/bin/python -m src.selection_checks --protocol iec104 --analysis rankings --overwrite
.venv/bin/python -m src.selection_checks --protocol iec104 --analysis matched-mi --overwrite
.venv/bin/python -m src.train_reduced_models --protocol iec104 --selector mi_matched --overwrite
.venv/bin/python -m src.evaluate_test --protocol iec104 --selector mi_matched --overwrite
.venv/bin/python -m src.benchmark_models --protocol iec104 --repeats 10 --seed 50 --overwrite
.venv/bin/python -m src.analyse_reduction --protocol iec104 --overwrite
.venv/bin/python -m src.plot_results --protocol iec104 --sensitivity
.venv/bin/python -m src.compare_results
```

`--overwrite` replaces outputs of the requested operation. The additional selector
uses its own output folders; it does not rerun the primary selectors or full-model
search. Changed selection settings require dependent reduced fitting and evaluation
to be repeated before their outputs are used. Failed or interrupted operations are
not accepted as completed inputs.

The output map is:

- `results/iec104/sensitivity/rank_agreement.csv`: pairwise fold rank correlations
  and selected-set overlap at both budgets.
- `results/iec104/sensitivity/field_selection_frequency.csv`: field inclusion in
  zero to three training folds. Fold populations overlap, so these are descriptive
  consistency measurements without independent-replication claims or p-values.
- `results/iec104/sensitivity/feature_selection/`: matched MI ranks, selected fields,
  actual sample/class support and settings. It uses 838 rows per fold-training scope
  and 873 all-development rows, compared with 7,002–7,003 and 8,504 for main MI.
- `results/iec104/sensitivity/validation/`: 40 new fits, saved-pipeline validation
  metrics, class changes and representation-overlap diagnostics.
- `results/iec104/sensitivity/test/`: ten new reduced conditions and six reproduced
  full/dummy references, predictions-linked errors and paired group intervals.
- `results/iec104/sensitivity/selector_comparisons.csv` and
  `selector_class_comparisons.csv`: matched MI minus each main selector at the same
  budget and population; full-model comparisons remain in the validation/test tables.
- `results/iec104/sensitivity/benchmark/`: ten-round raw prediction times, medians,
  quartiles and reduced/full ratios for all 36 unique pipelines. Loading and one
  complete warm-up per pipeline are excluded. Condition order is shuffled per round
  with seed 50; workers remain limited to two. External system load is not controlled.
- `data/iec104/sensitivity/`: actual matched sample identities and validation/test
  probabilities. These are local reusable data artefacts, not Git upload files.
- `models/iec104/reduced/mi_matched_25/` and `mi_matched_50/`: the 40 fitted pipelines.
- `figures/iec104/sensitivity/`: PNG consistency, sampling-comparison and repeated
  prediction figures, reproduced from numerical tables.

The consolidated IEC 104 prediction ratios use the common ten-round benchmark.
The three-repeat times stored in primary evaluation tables describe their own
execution; they are not substituted for the benchmark. Classifier fitting times
remain single recorded fits, with transformation measured separately. Selector
timers retain their declared boundaries, including SHAP contribution saving.
No comparison establishes end-to-end detection latency, energy use or equivalence.


## Dependent output validity and reduction diagnostics

Overwriting preparation, allocation, preprocessing, full fitting, selection,
reduced fitting or test evaluation invalidates the relevant downstream completion
records. Consolidation and plotting reject stale results. The consolidated CSV
and comparison PNG are regenerated only after their inputs are current. A ranking
agreement calculation alone does not invalidate models because it does not change
selected inputs. Fitted models and numerical files remain available locally, but
stale completion states cannot be presented as current results.

Before saved reduced models are used, the code checks their selected fields,
learned numeric replacements/scales, categorical encoder settings and categories,
column maps, class order and fitted parameters against the full recipe. Category
encoders are compared by their learned state, not object identity. Permitted LR
convergence retries are checked against recorded effective iteration ceilings and
attempt counts. These guards use actual saved artefacts; no artificial datasets
or test fixtures are required.

`src/analyse_reduction.py` adds two explanatory analyses without fitting or scoring:

- `results/iec104/sensitivity/diagnostics/representation_errors.csv` stratifies
  actual held-fold and test errors by whether the encoded vector occurs in the
  corresponding training population. It records each model's preprocessing view,
  float64 comparison precision, mixed-label training signatures and matching rows
  whose true label is absent from that signature's training labels. Undefined
  stratum error rates remain missing instead of being reported as zero.
- `representation_class_errors.csv` retains the same breakdown for all eight
  classes. Strata have different class compositions and are descriptive; record
  matching is not assumed to cause every error or automatically imply leakage.
- `cost_accounting.csv` reports measured selection/refitting overhead and the
  number of repeated identical prediction tasks needed to offset it, if a positive
  saving is observed. It includes single-classifier and five-classifier alternatives.
  The full panel and selected settings are assumed already available. Shared
  selector cost is charged once within each row; alternative rows must not be added.
  Actual operating volume, acquisition costs, full-panel creation, tuning, energy
  and retraining schedules are outside this accounting. It is not an operational
  break-even estimate or proof of equivalent detection performance.

`results/iec104/sensitivity/analysis.json` records the analysis inputs, definitions,
assumptions and completion state. New evaluations or benchmark outputs invalidate
this analysis until it is regenerated.

The ranking table also includes the expected random shared count k*k/p and the
chance-adjusted equal-size overlap (shared-k*k/p)/(k-k*k/p), discussed by Nogueira,
Sechidis and Brown (2018), https://jmlr.org/papers/v18/17-514.html . These are analytic
references, not generated random subsets or extra classifier conditions. The
three training folds overlap; the indices remain descriptive without stability
hypothesis tests or an optimal-budget verdict.


## IEC 61850 sampling, consistency and cost analyses

The packet comparison includes an MI ranking calculated on the exact saved SHAP
foreground observations. It uses the existing packet MI estimator, nominal
coding, training replacements and maximum-component field aggregation. This
controls observation identity relative to SHAP; it does not equate MI's empirical
class proportions with SHAP's equal-class attribution average. Comparison with
main MI changes sample size, selected identities and class composition together.
The main MI and common SHAP conditions remain separately reported.

The matched samples contain 591, 607 and 592 observations in fold-training scopes
0, 1 and 2, and 639 in all development data. Main MI uses 8,079, 8,095, 8,080 and
8,127 respectively. No test observations enter these rankings. Both budgets use
prepared fields, retaining all fitted columns of each selected field: 17 or 34
of 68 fields. Field reduction need not produce proportional encoded-width or
prediction-time reductions, particularly for payload and list fields.

Run the following from the project root using the existing environment:

```bash
.venv/bin/python -m src.selection_checks --protocol iec61850 --analysis matched-mi --overwrite
.venv/bin/python -m src.train_reduced_models --protocol iec61850 --selector mi_matched --overwrite
.venv/bin/python -m src.evaluate_test --protocol iec61850 --selector mi_matched --overwrite
.venv/bin/python -m src.benchmark_models --protocol iec61850 --repeats 10 --seed 50 --overwrite
.venv/bin/python -m src.analyse_reduction --protocol iec61850 --overwrite
.venv/bin/python -m src.plot_results --protocol iec61850 --sensitivity
.venv/bin/python -m src.plot_results --protocol iec61850
.venv/bin/python -m src.compare_results
```

The matched-MI command also regenerates fold-agreement outputs for all three
selectors. To regenerate only those descriptive comparisons, use
`.venv/bin/python -m src.selection_checks --protocol iec61850 --analysis rankings --overwrite`.
Primary SHAP contributions and full-model tuning are reused. Only the additional
40 reduced classifiers are fitted: five models, two budgets and four training
scopes. The new test evaluation includes ten reduced conditions and six reproduced
full/dummy references, giving 36 unique conditions across the packet comparison.

Outputs under `results/iec61850/sensitivity/` are:

- `feature_selection/`: matched ranks, selected fields, actual sample support,
  logical-component MI scores and completion/settings JSON. Component scores
  retain the limitation that wider fields offer more chances for a large maximum.
- `rank_agreement.csv`, `field_selection_frequency.csv` and
  `ranking_consistency.json`: all three fold-pair correlations, Jaccard and
  chance-adjusted agreement at both budgets. Overlapping folds provide descriptive
  within-development consistency rather than independent-data stability.
- `validation/`: fitting records, class/source changes and held-fold metrics.
- `test/`: class metrics, exact confusion counts, paired errors, all nine source
  results and uncertainty support. The 32 PTP test packets occupy one component;
  primary all-class intervals remain unsupported. Empty interval tables and the
  `not_estimable` state are intentional. No row bootstrap replaces whole groups.
- `selector_comparisons.csv` and `selector_class_comparisons.csv`: matched MI
  compared with each main selector on identical observations and budgets.
- `benchmark/`: 360 timed tasks across 36 pipelines and ten interleaved rounds.
  One full-task warm-up per pipeline, loading and verification are excluded.
  Reported prediction savings use median reduced time divided by median full time;
  medians of within-round ratios and their quartiles remain separate summaries.
- `diagnostics/representation_errors.csv` and `representation_class_errors.csv`:
  saved errors stratified by float64 encoded matching, training-label ambiguity
  and matching targets absent from training labels. Exact matching uses complete
  encoded row values; numeric signed zeros are treated equally.
- `diagnostics/representation_source_errors.csv`: the same error breakdown for
  each represented source and class, with source-specific dependency-group counts.
  For normal-labelled sources, the error rate is the false-alarm rate. Group
  counts can overlap across sources and must not be added as independent support.
- `diagnostics/cost_accounting.csv` and `analysis.json`: marginal selection/refit
  offsets for single-classifier and five-classifier alternatives. Full models and
  selected settings are assumed available; recorded selector and fitting times
  come from their respective executions. These are conditional measured-cost
  quantities, without deployment-volume, energy or operational-cost claims.

Actual sample identities and validation/test probabilities are retained under
`data/iec61850/sensitivity/`. The fitted models are under
`models/iec61850/reduced/mi_matched_25/` and `mi_matched_50/`. These remain local
artefacts excluded from Git. PNG figures are regenerated under
`figures/iec61850/sensitivity/`, and the existing validation/test normal-source
false-alarm figures include the added conditions. Main-normal, normal-PTP and
normal-fault errors remain separate because a pooled score can conceal poor
behaviour on a legitimate recording scenario.

These supplementary analyses were introduced after primary test outcomes were
available and provide exploratory sensitivity and diagnostic evidence. All
conditions are retained, including performance losses. They do not create a new
independent test population, identify causal attack measurements or establish
formal performance preservation. Preparation, sampled history, payload encoding
and group allocation remain fixed for the comparisons.

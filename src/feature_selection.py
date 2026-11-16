"""Shared training-row sampling, checked SHAP explanations and field rankings.

Signed encoded contributions are summed within each original field before
absolute values and class-balanced averaging. Consensus uses ranks so different
model-output scales are not averaged as if they were directly comparable.
The overall mean-rank idea builds on Arreche, Guntur and Abdallah (2026),
doi:10.1109/TON.2026.3731893; field/class aggregation is specified for this study.
"""

import os
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from src import settings
from src.checks import require
from src.shap_backend import precision_backend


def selection_recipe(mi_field_representation):
    """Keep the scientific choices separate from the resulting selected names."""
    return {
        "selectors": ["shap", "mi"], "retention_fractions": list(settings.RETENTION_FRACTIONS),
        "denominator": "active original fields in this training scope; round counts upward",
        "shap_foreground_cap_per_true_class": settings.SHAP_FOREGROUND_PER_CLASS,
        "shap_background_cap_per_true_class": settings.SHAP_BACKGROUND_PER_CLASS,
        "shap_sample_seed": settings.SELECTION_SAMPLE_SEED,
        "shap_sampling": "without replacement within each true class; all available when below cap",
        "shap_perturbation": "interventional with independent encoded components and explicit background",
        "shap_output": {name: "class_probability" if name in {"decision_tree", "random_forest"}
                        else "linear_decision_score" if name == "logistic_regression" else "class_margin"
                        for name in settings.CLASSIFIER_PANEL},
        "shap_field_aggregation": "signed component sum, then absolute value; mean across output classes, then within true class, then equal true-class mean",
        "shap_group_interpretation": "aggregated component attributions, not jointly masked coalition-group SHAP",
        "shap_common_rank": "equal model mean of average-tie ranks; exact doubled integer sums; alphabetical final ties",
        "additivity_atol": settings.SHAP_ADDITIVITY_ATOL, "additivity_rtol": settings.SHAP_ADDITIVITY_RTOL,
        "tree_precision_correction": "https://github.com/shap/shap/pull/5122; preserve float64 thresholds in SHAP 0.53.0",
        "mi_sample_cap_per_true_class": settings.MI_ROWS_PER_CLASS,
        "mi_sample_seed": settings.MI_SAMPLE_SEED, "mi_estimator_seed": settings.MI_ESTIMATION_SEED,
        "mi_neighbours": settings.MI_NEIGHBOURS,
        "mi_field_representation": mi_field_representation,
        "mi_distribution": "capped stratified training sample; residual class imbalance retained; no duplicated rare rows",
        "mi_ties": "average-tie ranks; alphabetical final ties",
        "reduced_parameter_policy": settings.REDUCED_PARAMETER_POLICY,
        "performance_interpretation": "paired observed differences; no formal equivalence claim or unapproved loss margin",
        "additional_conditions": "no PI selector, classifier-specific reduced SHAP fits, or frequency-versus-balanced SHAP experiment",
        "test_scoring": False,
    }


def stratified_rows(labels, maximum, seed, scope_index, role, class_count):
    """Select bounded real training rows, preserving every class without copying rows."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, scope_index, role]))
    positions = []
    for code in range(class_count):
        pool = np.flatnonzero(labels == code)
        require(len(pool) > 0, "A selection training population lacks a class.")
        positions.extend(rng.choice(pool, min(maximum, len(pool)), replace=False).tolist())
    return np.asarray(sorted(positions), dtype=int)


def field_attributions(values, processor, class_count):
    """Return signed contributions of whole original fields before taking magnitudes."""
    require(values.ndim == 3 and values.shape[1:] == (processor.n_features_out_, class_count),
            "Unexpected multiclass SHAP shape.")
    return np.stack([values[:, processor.field_columns([field]), :].sum(axis=1)
                     for field in processor.active_fields_], axis=1)


def raw_outputs(estimator, name, matrix):
    if name == "logistic_regression":
        return np.asarray(estimator.decision_function(matrix))
    if name == "xgboost":
        return np.asarray(estimator.predict(matrix, output_margin=True))
    if name == "lightgbm":
        return np.asarray(estimator.predict(matrix, raw_score=True))
    return np.asarray(estimator.predict_proba(matrix))


def shap_ranking(pipeline, foreground, background, labels, processor, destination, class_labels):
    """Explain actual training rows and reject explanations that do not reconstruct outputs."""
    # Keep plotting's optional font cache in the writable, ignored local cache.
    os.environ.setdefault("MPLCONFIGDIR", str(settings.PROJECT_ROOT / ".cache/matplotlib"))
    import shap

    name = pipeline.model_name
    started = perf_counter()
    x = processor.transform_for_model(foreground, name)
    reference = processor.transform_for_model(background, name)
    original = raw_outputs(pipeline.estimator, name, x)
    dtype = np.float32 if name in {"decision_tree", "random_forest", "xgboost"} else np.float64
    x, reference = x.astype(dtype), reference.astype(dtype)
    require(np.isfinite(x).all() and np.isfinite(reference).all(), "Nonfinite native SHAP inputs.")
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=settings.TRAINING_WORKERS):
        warnings.simplefilter("always")
        masker = shap.maskers.Independent(reference, max_samples=len(reference))
        require(np.array_equal(masker.data, reference), "SHAP silently changed its background sample.")
        if name == "logistic_regression":
            explainer = shap.LinearExplainer(pipeline.estimator, masker)
            explanation = explainer(x)
            build = None
        else:
            with precision_backend() as build:
                explainer = shap.TreeExplainer(pipeline.estimator, data=masker,
                                               feature_perturbation="interventional", model_output="raw")
                explanation = explainer(x)
        values = np.asarray(explanation.values, dtype=np.float64)
        base = np.asarray(explanation.base_values, dtype=np.float64)
        outputs = raw_outputs(pipeline.estimator, name, x)
    require(np.allclose(outputs, original, atol=1e-12, rtol=1e-12), "Native input conversion changed model outputs.")
    require(np.isfinite(values).all() and np.isfinite(base).all(), "Nonfinite SHAP contributions.")
    reconstructed = values.sum(axis=1) + base
    require(np.allclose(reconstructed, outputs, atol=settings.SHAP_ADDITIVITY_ATOL,
                        rtol=settings.SHAP_ADDITIVITY_RTOL), f"Independent SHAP additivity check failed: {name}")
    grouped = field_attributions(values, processor, len(class_labels))
    require(np.allclose(grouped.sum(axis=1) + base, outputs,
                        atol=settings.SHAP_ADDITIVITY_ATOL, rtol=settings.SHAP_ADDITIVITY_RTOL),
            "Original-field grouping lost contributions.")
    magnitude = np.abs(grouped).mean(axis=2)
    by_class = np.stack([magnitude[labels == c].mean(axis=0) for c in range(len(class_labels))])
    scores = by_class.mean(axis=0)
    table = pd.DataFrame({"field": processor.active_fields_, "importance": scores,
                          "rank": rankdata(-scores, method="average"), "method": "shap_model", "model": name})
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, encoded_values=values, field_values=grouped, base_values=base,
                        model_outputs=outputs, foreground_development_positions=foreground.index.to_numpy(dtype=np.int64))
    class_rows = [{"model": name, "true_class": label, "field": field, "importance": float(by_class[c, f])}
                  for c, label in enumerate(class_labels) for f, field in enumerate(processor.active_fields_)]
    info = {"model": name, "seconds_including_transform_and_save": perf_counter() - started,
            "maximum_additivity_absolute_error": float(np.abs(reconstructed - outputs).max()),
            "foreground_rows": len(x), "background_rows": len(reference), "native_input_dtype": str(np.dtype(dtype)),
            "precision_backend": build,
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
    return table, class_rows, info


def consensus_table(rankings):
    """Combine half-integer ranks exactly, avoiding spurious floating-point tie breaks."""
    fields = sorted(next(iter(rankings.values())).field)
    table = pd.DataFrame({"field": fields})
    total = np.zeros(len(fields), dtype=np.int64)
    for name in settings.CLASSIFIER_PANEL:
        ranking = rankings[name].set_index("field")
        require(ranking.index.is_unique and set(ranking.index) == set(fields), "SHAP field universes differ.")
        ranks = ranking.loc[fields, "rank"].to_numpy()
        doubled = np.rint(2 * ranks).astype(np.int64)
        require(np.allclose(doubled, 2 * ranks, atol=1e-12, rtol=0), "Non-half-integer feature ranks.")
        table[name + "_rank"] = ranks
        total += doubled
    table["twice_rank_sum"] = total
    table["mean_rank"] = total / (2 * len(settings.CLASSIFIER_PANEL))
    table = table.sort_values(["twice_rank_sum", "field"]).reset_index(drop=True)
    table["selection_order"] = np.arange(1, len(table) + 1)
    return table



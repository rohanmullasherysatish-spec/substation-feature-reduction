"""Describe fold consistency and calculate MI on saved SHAP foreground rows.

Both operations use development data only. The matched MI condition changes
the observations assessed by MI, while preserving its estimator and field rules.
"""

import argparse
import importlib.metadata
from itertools import combinations
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import settings
from src.checks import require, invalidate_results
from src.evaluate_validation import full_record
from src.train_full_models import atomic_json, load_training_inputs


def protocol_selector(protocol):
    if protocol == "iec104":
        from src.iec104 import feature_selection
    elif protocol == "iec61850":
        from src.iec61850 import feature_selection
    else:
        raise ValueError("Unknown protocol.")
    return feature_selection


def matched_recipe(protocol):
    return {
        "selector": "mi_matched",
        "reference_selection_recipe": protocol_selector(protocol).selection_recipe(),
        "sampling": "exact saved SHAP foreground identities in each training scope; no resampling or duplicated rows",
        "sample_role": "mi_matched",
        "estimator_seed": settings.MI_ESTIMATION_SEED,
        "neighbours": settings.MI_NEIGHBOURS,
        "retention_fractions": list(settings.RETENTION_FRACTIONS),
        "field_policy": "reuse the protocol MI estimator, discrete flags, training replacements and field aggregation",
        "interpretation": "sampling sensitivity of configured selectors; matching rows does not match class aggregation or attribution targets",
        "test_records_used": 0,
    }


def ranking_consistency(root=settings.PROJECT_ROOT, protocol="iec104", overwrite=False):
    """Compare overlapping training folds descriptively, without significance tests."""
    root = Path(root)
    source = root / f"results/{protocol}/feature_selection"
    destination = root / f"results/{protocol}/sensitivity"
    names = ("rank_agreement.csv", "field_selection_frequency.csv", "ranking_consistency.json")
    require(overwrite or not any((destination / name).exists() for name in names),
            "Ranking comparisons exist; use --overwrite deliberately.")
    status = json.loads((source / "selection.json").read_text())
    require(status["status"] == "complete"
            and status["recipe"] == protocol_selector(protocol).selection_recipe(),
            "Main feature selection is incomplete or settings changed.")
    scopes = [f"fold_{i}" for i in range(settings.VALIDATION_FOLDS)]
    common = pd.read_csv(source / "common_ranking.csv")
    individual = pd.read_csv(source / "rankings.csv")
    tables = {"shap": common.rename(columns={"mean_rank": "rank"}),
              "mi": individual.loc[individual.method.eq("mi")].copy()}
    extra = destination / "feature_selection"
    if (extra / "selection.json").exists():
        record = json.loads((extra / "selection.json").read_text())
        require(record["status"] == "complete" and record["recipe"] == matched_recipe(protocol),
                "Matched MI selection is incomplete or settings changed.")
        tables["mi_matched"] = pd.read_csv(extra / "rankings.csv")
    # An interrupted calculation must not leave an older completion marker.
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rank_agreement_", dir=root) as folder:
        atomic_json(destination / names[2], {"status": "running", "protocol": protocol}, Path(folder))
    agreement, frequencies = [], []
    for selector, table in tables.items():
        folds = {scope: table.loc[table.scope.eq(scope)].copy() for scope in scopes}
        universe = set(folds[scopes[0]].field)
        require(universe and all(len(t) == len(universe) and set(t.field) == universe
                                 and t.field.is_unique and np.isfinite(t["rank"]).all()
                                 for t in folds.values()), "Fold ranking field universes differ.")
        ordered = {scope: (t.sort_values("selection_order") if selector == "shap"
                           else t.sort_values(["rank", "field"])) for scope, t in folds.items()}
        for fraction in settings.RETENTION_FRACTIONS:
            count = math.ceil(fraction * len(universe))
            chosen = {scope: set(t.field.iloc[:count]) for scope, t in ordered.items()}
            for left, right in combinations(scopes, 2):
                a = folds[left].set_index("field")["rank"].sort_index()
                b = folds[right].set_index("field")["rank"].reindex(a.index)
                # A constant ranking has undefined correlation, not zero agreement.
                correlation = float(spearmanr(a, b).statistic) if a.nunique() > 1 and b.nunique() > 1 else np.nan
                shared = len(chosen[left] & chosen[right])
                expected_shared = count * count / len(universe)
                # Equal-sized subsets permit analytic correction for random overlap.
                adjusted = ((shared - expected_shared) / (count - expected_shared)
                            if count < len(universe) else np.nan)
                agreement.append({"selector": selector, "fold_a": left, "fold_b": right,
                    "active_fields": len(universe), "retention_fraction": fraction,
                    "selected_fields": count, "shared_fields": shared,
                    "jaccard": shared / len(chosen[left] | chosen[right]),
                    "expected_shared_random": expected_shared,
                    "chance_adjusted_overlap": adjusted,
                    "spearman": correlation})
            for field in sorted(universe):
                times = sum(field in chosen[scope] for scope in scopes)
                frequencies.append({"selector": selector, "retention_fraction": fraction,
                    "field": field, "selected_folds": times, "folds": len(scopes),
                    "selection_frequency": times / len(scopes)})
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agreement).to_csv(destination / names[0], index=False)
    pd.DataFrame(frequencies).to_csv(destination / names[1], index=False)
    (destination / names[2]).write_text(json.dumps({"status": "complete", "protocol": protocol,
        "selectors": list(tables), "folds": scopes, "retention_fractions": list(settings.RETENTION_FRACTIONS),
        "rank_correlation": "Spearman with tied ranks; undefined for constant rankings",
        "subset_ties": "main selection order; average ranks then alphabetical ties for MI",
        "chance_adjustment": "Kuncheva equal-size index: (shared - k*k/p)/(k-k*k/p); undefined when k=p",
        "chance_reference": "Nogueira, Sechidis and Brown (2018), JMLR 18(174), 1-54; https://jmlr.org/papers/v18/17-514.html",
        "interpretation": "descriptive agreement between overlapping training folds; not independent resampling stability",
        "models_fitted": 0, "test_records_used": 0}, indent=2) + "\n")
    summary = pd.DataFrame(agreement).groupby(["selector", "retention_fraction"])[["shared_fields", "jaccard", "chance_adjusted_overlap", "spearman"]].mean()
    print(summary.to_string(), flush=True)
    print("Saved fold-ranking comparisons; no fitting or test scoring.", flush=True)


def matched_mi(root=settings.PROJECT_ROOT, protocol="iec104", overwrite=False):
    """Use the same actual rows as SHAP, not merely the same per-class cap."""
    root = Path(root)
    module = protocol_selector(protocol)
    results = root / f"results/{protocol}/sensitivity/feature_selection"
    local = root / f"data/{protocol}/sensitivity/feature_selection"
    names = ("rankings.csv", "selected_features.csv", "sample_support.csv", "selection.json")
    require(overwrite or not any((results / n).exists() for n in names),
            "Matched MI outputs exist; use --overwrite deliberately.")
    reference = json.loads((root / f"results/{protocol}/feature_selection/selection.json").read_text())
    require(reference["status"] == "complete" and reference["recipe"] == module.selection_recipe(),
            "Reference selection is incomplete or settings changed.")
    record = full_record(root, protocol)
    for name, version in reference["packages"].items():
        require(importlib.metadata.version(name) == version, f"Selection dependency changed: {name}")
    inputs, labels, scopes, processors = load_training_inputs(root, protocol)
    metadata = pd.read_parquet(root / f"data/{protocol}/assignments.parquet")
    metadata = metadata.loc[metadata.partition.eq("development")].reset_index(drop=True)
    stored = pd.read_parquet(root / f"data/{protocol}/feature_selection/samples.parquet")
    identities = ["record_id", "dataset_row", "source_file", "source_row", "group_id", "class_label"]
    require(set(stored.scope) == set(scopes), "Saved sample scope coverage differs.")
    results.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "protocol": protocol, "recipe": matched_recipe(protocol),
        "completed_scopes": [], "analyses": [], "full_training_recipe": record["recipe"],
        "selected_full_parameters": {name: record["selected"][name]["parameters"] for name in settings.CLASSIFIER_PANEL},
        "packages": dict(reference["packages"]), "classifiers_fitted": 0, "test_records_used": 0}
    rankings, subsets, samples, support = [], [], [], []
    classes = settings.protocol_settings(protocol).CLASSES
    with TemporaryDirectory(prefix=f".selection_{protocol}_", dir=root) as folder:
        staging = Path(folder)
        invalidate_results(root, protocol, "selection", "mi_matched")
        atomic_json(results / "selection.json", status, staging)
        try:
            for scope, mask in scopes.items():
                foreground = stored.loc[stored.scope.eq(scope) & stored.role.eq("foreground")].copy()
                positions = foreground.development_position.to_numpy(dtype=int)
                require(len(positions) > 0 and len(np.unique(positions)) == len(positions)
                        and (positions >= 0).all() and (positions < len(inputs)).all()
                        and mask[positions].all(), "Foreground identities cross the training boundary.")
                require(metadata.iloc[positions][identities].reset_index(drop=True).equals(
                    foreground[identities].reset_index(drop=True)), "Saved foreground identities differ.")
                table, *detail, info = module.mi_ranking(inputs.iloc[positions], labels[positions], processors[scope])
                table = table.assign(method="mi_matched", scope=scope)
                rankings.append(table)
                foreground["role"] = "mi_matched"
                samples.append(foreground)
                original = stored.loc[stored.scope.eq(scope) & stored.role.eq("mi")]
                for code, label in enumerate(classes):
                    n = int((labels[positions] == code).sum())
                    require(n == min(settings.SHAP_FOREGROUND_PER_CLASS, int((labels[mask] == code).sum())),
                            "Foreground sample class support differs from the declared cap.")
                    support.append({"scope": scope, "class_label": label,
                        "training_rows": int((labels[mask] == code).sum()), "matched_rows": n,
                        "original_mi_rows": int(original.class_label.eq(label).sum()),
                        "matched_class_proportion": n / len(positions),
                        "shap_true_class_aggregation_weight": 1 / len(classes)})
                ordered = table.sort_values(["rank", "field"]).field.tolist()
                for fraction in settings.RETENTION_FRACTIONS:
                    count = math.ceil(fraction * len(ordered))
                    subsets.extend({"scope": scope, "selector": "mi_matched", "retention_fraction": fraction,
                        "active_original_fields": len(ordered), "selected_field_count": count,
                        "selection_order": i + 1, "field": field} for i, field in enumerate(ordered[:count]))
                if detail:
                    detail[0].assign(scope=scope).to_csv(staging / f"components_{scope}.csv", index=False)
                status["analyses"].append(dict(info, method="mi_matched", scope=scope))
                status["completed_scopes"].append(scope)
                atomic_json(results / "selection.json", status, staging)
                print(f"{scope}: matched MI on {len(positions):,} saved foreground rows; no test rows.", flush=True)
            for name, frame in (("rankings.csv", pd.concat(rankings, ignore_index=True)),
                                ("selected_features.csv", pd.DataFrame(subsets)),
                                ("sample_support.csv", pd.DataFrame(support))):
                frame.to_csv(staging / name, index=False)
                (staging / name).replace(results / name)
            pd.concat(samples, ignore_index=True).to_parquet(local / "samples.parquet", index=False)
            if detail:
                pd.concat([pd.read_csv(staging / f"components_{scope}.csv") for scope in scopes], ignore_index=True).to_csv(results / "mi_components.csv", index=False)
            status["status"] = "complete"
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(results / "selection.json", status, staging)
    ranking_consistency(root, protocol, overwrite=True)


def selector_comparisons(root=settings.PROJECT_ROOT, protocol="iec104"):
    """Compare the added MI condition with both main selectors on identical rows."""
    root = Path(root)
    base = root / f"results/{protocol}"
    extra = base / "sensitivity"
    differences, class_differences = [], []
    for population in ("validation", "test"):
        metric_name = "reduced_selected_metrics.csv" if population == "validation" else "point_metrics.csv"
        class_name = "reduced_class_metrics.csv" if population == "validation" else "class_metrics.csv"
        reference = pd.read_csv(base / population / metric_name)
        current = pd.read_csv(extra / population / metric_name)
        reference_classes = pd.read_csv(base / population / class_name)
        current_classes = pd.read_csv(extra / population / class_name)
        if population == "test":
            for frame in (reference, current, reference_classes, current_classes):
                frame["scope"] = "test"
        else:
            reference_classes = reference_classes.loc[reference_classes.population.eq("validation")]
            current_classes = current_classes.loc[current_classes.population.eq("validation")]
        ref_index = reference.set_index(["model", "condition", "scope"])
        ref_class_index = reference_classes.set_index(["model", "condition", "scope", "class_label"])
        for row in current.loc[current.condition.str.startswith("mi_matched_")].to_dict("records"):
            percentage = row["condition"].rsplit("_", 1)[1]
            for method in ("mi", "shap"):
                reference_condition = f"{method}_{percentage}"
                old = ref_index.loc[row["model"], reference_condition, row["scope"]]
                require(row["records"] == old.records, "Selector comparison population sizes differ.")
                item = {"population": population, "scope": row["scope"], "model": row["model"],
                        "condition": row["condition"], "reference_condition": reference_condition,
                        "records": int(row["records"])}
                for metric in ("macro_f1", "accuracy", "weighted_f1", "normal_false_alarm_rate", "errors"):
                    item.update({"reference_"+metric: old[metric], "matched_"+metric: row[metric],
                                 "difference_"+metric: row[metric]-old[metric]})
                differences.append(item)
                subset = current_classes.loc[current_classes.model.eq(row["model"])
                    & current_classes.condition.eq(row["condition"]) & current_classes.scope.eq(row["scope"])]
                require(len(subset) == len(settings.protocol_settings(protocol).CLASSES), "Class comparison coverage differs.")
                for cls in subset.to_dict("records"):
                    old_cls = ref_class_index.loc[row["model"], reference_condition, row["scope"], cls["class_label"]]
                    require(cls["support"] == old_cls.support, "Selector class supports differ.")
                    pair = {key: item[key] for key in ("population", "scope", "model", "condition", "reference_condition")}
                    pair.update(class_label=cls["class_label"], support=int(cls["support"]))
                    for metric in ("precision", "recall", "f1", "errors"):
                        pair.update({"reference_"+metric: old_cls[metric], "matched_"+metric: cls[metric],
                                     "difference_"+metric: cls[metric]-old_cls[metric]})
                    class_differences.append(pair)
    pd.DataFrame(differences).to_csv(extra / "selector_comparisons.csv", index=False)
    pd.DataFrame(class_differences).to_csv(extra / "selector_class_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), required=True)
    parser.add_argument("--analysis", choices=("rankings", "matched-mi"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    operation = ranking_consistency if args.analysis == "rankings" else matched_mi
    operation(protocol=args.protocol, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

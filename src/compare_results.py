"""Consolidate within-protocol test trade-offs from saved results; no predictions.

Every reduced condition remains in the table. Different class populations are
kept separate; scores are not pooled into a cross-protocol average.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings
from src.checks import require, validate_evaluation, validate_benchmark
from src.plot_results import NAMES, plt, save


def compare(root=settings.PROJECT_ROOT):
    root = Path(root)
    rows = []
    timing_descriptions = {}
    expected_rows = 0
    for protocol in ("iec104", "iec61850"):
        benchmark_path = root / f"results/{protocol}/sensitivity/benchmark/prediction_summary.csv"
        benchmark = None
        if benchmark_path.exists():
            status = validate_benchmark(root, protocol)
            benchmark = pd.read_csv(benchmark_path).set_index(["model", "condition"])
            timing_descriptions[protocol] = f"{status['repeats']} interleaved timing rounds"
        else:
            timing_descriptions[protocol] = f"{settings.PREDICTION_TIMING_REPEATS} consecutive evaluation repetitions"
        for role, suffix in (("main", ""), ("sampling_sensitivity", "sensitivity/")):
            result_root = root / f"results/{protocol}/{suffix}"
            folder = result_root / "test"
            if role != "main" and not (folder / "evaluation.json").exists():
                continue
            evaluation, _ = validate_evaluation(root, protocol, "mi_matched" if suffix else None)
            require(evaluation["status"] == "complete", "Complete the test evaluations first.")
            points = pd.read_csv(folder / "point_metrics.csv").set_index(["model", "condition"])
            pairs = pd.read_csv(folder / "paired_comparisons.csv")
            classes = pd.read_csv(folder / "class_paired_comparisons.csv")
            selection = json.loads((result_root / "feature_selection/selection.json").read_text())
            expected_pairs = len(settings.CLASSIFIER_PANEL)*(len(evaluation["plan"]["conditions"])-1)
            require(selection["status"] == "complete" and len(pairs) == expected_pairs
                    and len(points) == expected_pairs + len(settings.CLASSIFIER_PANEL) + 1,
                    "Unexpected completed comparison coverage.")
            expected_rows += expected_pairs
            for pair in pairs.itertuples():
                model, condition = pair.model, pair.condition
                selector, percent = condition.rsplit("_", 1)
                full, reduced = points.loc[model, "full"], points.loc[model, condition]
                require(full.records == reduced.records, "Paired test populations differ.")
                np.testing.assert_allclose(reduced.macro_f1-full.macro_f1, pair.difference_macro_f1,
                                           atol=1e-12, rtol=1e-12)
                subset = classes[(classes.model == model) & (classes.condition == condition)]
                worst = subset.sort_values(["difference_recall", "class_label"]).iloc[0]
                analyses = [a for a in selection["analyses"] if
                            (a.get("model") in settings.CLASSIFIER_PANEL if selector == "shap"
                             else a.get("method") == selector)]
                def seconds(analysis):
                    return analysis.get("seconds_including_transform_and_save",
                                        analysis.get("seconds_including_transform", 0))
                row = {"protocol": protocol, "model": model, "condition": condition,
                       "comparison_role": role, "selector": selector, "retention_fraction": int(percent)/100,
                       "test_records": int(full.records), "test_groups": evaluation["test_groups"],
                       "full_macro_f1": full.macro_f1, "reduced_macro_f1": reduced.macro_f1,
                       "difference_macro_f1": pair.difference_macro_f1,
                       "full_original_fields": int(full.active_original_fields),
                       "reduced_original_fields": int(reduced.active_original_fields),
                       "full_encoded_columns": int(full.encoded_columns),
                       "reduced_encoded_columns": int(reduced.encoded_columns),
                       "full_normal_false_alarm_rate": full.normal_false_alarm_rate,
                       "reduced_normal_false_alarm_rate": reduced.normal_false_alarm_rate,
                       "difference_errors": int(pair.difference_errors),
                       "worst_recall_change_class": worst.class_label,
                       "worst_recall_change_class_support": int(worst.support),
                       "minimum_class_recall_change": worst.difference_recall,
                       "primary_uncertainty_status": evaluation["uncertainty"]["status"],
                       "executed_group_resamples": evaluation["uncertainty"]["executed_replicates"],
                       "shared_selector_seconds_all_scopes": sum(seconds(a) for a in analyses),
                       "shared_selector_seconds_development": sum(seconds(a) for a in analyses
                                                                   if a["scope"] == "development")}
                for column, label, ratio in (("development_fit_seconds", "fit_seconds", "fit"),
                                             ("pipeline_prediction_seconds", "prediction_seconds", "prediction"),
                                             ("pipeline_bytes", "pipeline_bytes", "pipeline_bytes")):
                    before, after = full[column], reduced[column]
                    if ratio == "prediction" and benchmark is not None:
                        before = benchmark.loc[model, "full"]["median_seconds"]
                        after = benchmark.loc[model, condition]["median_seconds"]
                    require(before > 0 and after > 0, "Invalid recorded cost.")
                    row[f"full_{label}"] = before
                    row[f"reduced_{label}"] = after
                    row[f"{ratio}_ratio"] = after/before
                row["prediction_timing_source"] = ("interleaved warm-process benchmark" if benchmark is not None
                                                   else "consecutive evaluation repetitions")
                rows.append(row)
    table = pd.DataFrame(rows)
    require(len(table) == expected_rows and not table.duplicated(["protocol", "model", "condition"]).any(),
            "Incomplete consolidated comparison.")
    destination = root / "results/comparison/reduction_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.88, wspace=.18)
    colours = dict(zip(settings.CLASSIFIER_PANEL, plt.get_cmap("tab10").colors))
    for ax, protocol in zip(axes, ("iec104", "iec61850")):
        for model in settings.CLASSIFIER_PANEL:
            for selector, marker in (("shap", "o"), ("mi", "s"), ("mi_matched", "^")):
                subset = table[(table.protocol == protocol) & (table.model == model)
                               & (table.selector == selector)]
                for row in subset.itertuples():
                    # Fill encodes budget without overlapping text or moving data.
                    ax.scatter(row.prediction_ratio, row.difference_macro_f1,
                               s=65, marker=marker, edgecolors=colours[model], linewidths=1.6,
                               facecolors=colours[model] if row.retention_fraction == .25 else "none",
                               alpha=.85)
        ax.axhline(0, color="grey", linestyle="--")
        ax.axvline(1, color="grey", linestyle=":")
        ax.set(title="IEC 104" if protocol == "iec104" else "IEC 61850",
               xlabel="Test pipeline prediction time: reduced / full",
               ylabel="Test macro F1: reduced minus full")
        ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=colour, label=NAMES[model])
               for model, colour in colours.items()]
    handles += [Line2D([], [], marker=marker, linestyle="", color="black", label=label)
                for marker,label in (("o", "SHAP"), ("s", "MI"), ("^", "MI on SHAP rows"))]
    handles += [Line2D([], [], marker="o", linestyle="", color="black", markerfacecolor=face,
                       label=label) for face, label in (("black", "25% retained"), ("none", "50% retained"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .07), ncol=5, fontsize=8)
    fig.suptitle("Within-protocol trade-offs: every declared reduced condition")
    timing_note = "; ".join(("IEC 104" if p == "iec104" else "IEC 61850") + ": " + description
                            for p, description in timing_descriptions.items())
    fig.text(.5, .025, "Percentages refer to active prepared fields. Point estimates do not establish equivalence.\n" + timing_note + ".",
             ha="center", fontsize=9)
    save(fig, root / "figures/comparison", "performance_prediction_tradeoffs")
    print(f"Saved {len(table)} within-protocol comparisons and one PNG; no fitting, predictions or pooled protocol score.")


if __name__ == "__main__":
    compare()

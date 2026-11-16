"""Reproduce scientific figures from saved numerical outputs; never fit models."""

import argparse
import json
import os
from pathlib import Path

from src import settings
from src.checks import complete_record, validate_evaluation, validate_benchmark

os.environ.setdefault("MPLCONFIGDIR", str(settings.PROJECT_ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree",
         "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "dummy": "Majority reference"}
CLASS_NAMES = {"dos": "DoS", "flood": "Flood", "fuzzy": "Fuzzy", "iec104_starvation": "Starvation",
               "mitm": "MITM", "normal": "Normal", "ntp_ddos": "NTP DDoS", "port_scan": "Port scan",
               "ptp_attack": "PTP attack", "replay": "Replay"}


def save(figure, directory, name, dpi=180):
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / (name + ".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def validation_figures(root, protocol="iec104"):
    if protocol not in {"iec104", "iec61850"}:
        raise ValueError("Unknown protocol.")
    class_labels = settings.protocol_settings(protocol).CLASSES
    results = root / f"results/{protocol}/validation"
    record = json.loads((results / "prediction_recovery.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Prediction recovery is incomplete.")
    directory = root / f"figures/{protocol}/validation"
    metrics = pd.read_csv(results / "full_selected_metrics.csv")
    models = [*settings.CLASSIFIER_PANEL, "dummy"]
    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    for i, name in enumerate(models):
        folds = metrics.loc[(metrics.model == name) & metrics.scope.str.startswith("fold_"), "macro_f1"].to_numpy()
        ax.scatter(i + np.linspace(-.12, .12, len(folds)), folds, s=35, color="#23778b")
        ax.plot([i - .21, i + .21], [folds.mean()] * 2, color="#bd5727", linewidth=2.5)
    ax.set(xticks=range(len(models)), xticklabels=[NAMES[n] for n in models], ylabel="Validation macro F1",
           ylim=(0, 1.03), title="Full-input models: three validation folds")
    ax.text(.01, .02, "Dots: folds; orange line: fold mean. Development selection results.", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=.25)
    save(fig, directory, "full_macro_f1")

    classes = pd.read_csv(results / "full_selected_class_metrics.csv")
    pooled = classes[classes.scope == "pooled_validation"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    for ax, metric in zip(axes, ("precision", "recall")):
        values = pooled.pivot(index="model", columns="class_label", values=metric).loc[models, list(class_labels)]
        image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set(xticks=range(len(class_labels)), xticklabels=[CLASS_NAMES[c] for c in class_labels], yticks=range(len(models)),
               yticklabels=[NAMES[n] for n in models], title="Pooled held-fold " + metric)
        plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
        for i in range(len(models)):
            for j in range(len(class_labels)):
                v = values.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > .65 else "black", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=.8)
    fig.suptitle("Pooled validation: values rounded to two decimals; inspect confusion counts for exact errors")
    save(fig, directory, "full_class_precision_recall")

    confusions = pd.read_csv(results / "full_confusion_matrices.csv")
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained")
    for ax, name in zip(axes.flat, settings.CLASSIFIER_PANEL):
        table = confusions[(confusions.model == name) & (confusions.scope == "pooled_validation")]
        counts = table.pivot(index="true_class", columns="predicted_class", values="records").loc[list(class_labels), list(class_labels)].to_numpy()
        proportions = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(len(class_labels)), xticklabels=[CLASS_NAMES[c] for c in class_labels],
               yticks=range(len(class_labels)), yticklabels=[CLASS_NAMES[c] for c in class_labels],
               xlabel="Predicted class", ylabel="True class", title=NAMES[name])
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        for i, j in np.argwhere(counts > 0):
            ax.text(j, i, f"{counts[i,j]:,}", ha="center", va="center", fontsize=7,
                    color="white" if proportions[i, j] > .6 else "black")
    axes.flat[-1].axis("off")
    fig.suptitle("Full-input pooled validation confusions: counts; colour normalised within true class")
    save(fig, directory, "full_confusions")


def selection_figures(root, protocol="iec104"):
    results = root / f"results/{protocol}/feature_selection"
    if not (results / "selection.json").exists():
        return
    if json.loads((results / "selection.json").read_text())["status"] != "complete":
        return
    common = pd.read_csv(results / "common_ranking.csv")
    final = common[common.scope == "development"].sort_values("selection_order").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.barh(final.field, final.mean_rank, color="#23778b")
    ax.set(xlabel="Mean rank across five classifiers (smaller is more important)",
           title="Common SHAP ranking: all-development training analysis")
    ax.grid(axis="x", alpha=.2)
    save(fig, root / f"figures/{protocol}/feature_selection", "common_shap_ranking")


def reduction_figures(root, protocol="iec104"):
    classes = settings.protocol_settings(protocol).CLASSES
    results = root / f"results/{protocol}/validation"
    path = results / "reduced_training.json"
    if not path.exists():
        return
    validate_evaluation(root, protocol)
    full = pd.read_csv(results / "full_selected_metrics.csv")
    reduced = pd.read_csv(results / "reduced_selected_metrics.csv")
    full = full[full.scope.str.startswith("fold_")]
    reduced = reduced[reduced.scope.str.startswith("fold_")]
    directory = root / f"figures/{protocol}/reduction"
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained", sharey=True)
    for ax, name in zip(axes.flat, settings.CLASSIFIER_PANEL):
        baseline = full[full.model == name].macro_f1.mean()
        for selector, colour in (("shap", "#23778b"), ("mi", "#bd5727")):
            group = reduced[(reduced.model == name) & reduced.condition.str.startswith(selector + "_")]
            means = group.groupby("retention_fraction").macro_f1.mean().sort_index()
            ax.plot([*means.index, 1.0], [*means, baseline], "o-", color=colour, label=selector.upper())
            ax.scatter(group.retention_fraction, group.macro_f1, s=12, color=colour, alpha=.45)
        ax.set(title=NAMES[name], xticks=[.25, .5, 1], xticklabels=["25%", "50%", "Full"],
               xlabel="Retained prepared fields", ylabel="Validation macro F1", ylim=(0, 1.03))
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Full versus reduced models: lines show fold means; small dots show three folds")
    save(fig, directory, "validation_macro_f1")

    pairs = pd.read_csv(results / "class_reduction_comparisons.csv")
    pairs = pairs[pairs.scope == "pooled_validation"].copy()
    pairs["row"] = pairs.model.map(NAMES) + " / " + pairs.condition.str.replace("_", " ").str.upper() + "%"
    values = pairs.pivot(index="row", columns="class_label", values="difference_recall")[list(classes)]
    extent = max(.05, float(np.abs(values.to_numpy()).max()))
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    image = ax.imshow(values, vmin=-extent, vmax=extent, cmap="RdBu", aspect="auto")
    ax.set(xticks=range(len(classes)), xticklabels=[CLASS_NAMES[c] for c in classes],
           yticks=range(len(values)), yticklabels=values.index,
           title="Class recall change: reduced minus full, pooled validation")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(values)):
        for j in range(len(classes)):
            v = values.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > .65 * extent else "black")
    fig.colorbar(image, ax=ax, label="Recall difference", shrink=.7)
    save(fig, directory, "class_recall_change")

    # These are recorded model costs, not peak-memory or controlled deployment measurements.
    fits = pd.read_csv(results / "reduced_models.csv")
    full_fits = pd.read_csv(results / "full_models.csv")
    chosen = json.loads((results / "selected_parameters.json").read_text())["selected"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for name in settings.CLASSIFIER_PANEL:
        group = fits[(fits.model == name) & (fits.scope == "development")].set_index("condition")
        old = full_fits[(full_fits.model == name) & (full_fits.scope == "development")].iloc[0]
        old_size = (group.full_pipeline_bytes.iloc[0] if "full_pipeline_bytes" in group
                    else (root / chosen[name]["pipelines"]["development"]).stat().st_size)
        positions = np.arange(4)
        order = ["shap_25", "shap_50", "mi_25", "mi_50"]
        axes[0].plot(positions, group.loc[order, "fit_seconds"] / old.fit_seconds, "o-", label=NAMES[name])
        sizes = (group.loc[order, "pipeline_bytes"].to_numpy() if "pipeline_bytes" in group
                 else [(root / f"models/{protocol}/reduced/{condition}/{name}/development.joblib").stat().st_size for condition in order])
        axes[1].plot(positions, np.asarray(sizes) / old_size, "o-", label=NAMES[name])
        old_time = full[full.model == name].pipeline_prediction_seconds.sum()
        times = reduced[reduced.model == name].groupby("condition").pipeline_prediction_seconds.sum()
        axes[2].plot(positions, times.loc[order] / old_time, "o-", label=NAMES[name])
    for ax, title in zip(axes, ("All-development fitting time", "Saved development pipeline size", "Validation pipeline prediction time")):
        ax.axhline(1, color="grey", linestyle="--")
        ax.set(xticks=range(4), xticklabels=["SHAP 25%", "SHAP 50%", "MI 25%", "MI 50%"],
               ylabel="Reduced / full", title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle("Recorded computation ratios: descriptive local timings and pipeline sizes")
    save(fig, directory, "computation_ratios")


def source_error_figures(root, protocol="iec61850", population="validation"):
    """Keep the normal-fault weakness visible beside aggregate performance."""
    folder = root / f"results/{protocol}/{population}"
    status = folder / ("evaluation.json" if population == "test" else "reduced_training.json")
    source = folder / ("source_paired_comparisons.csv" if population == "test" else "source_reduction_comparisons.csv")
    if not status.exists() or json.loads(status.read_text())["status"] != "complete" or not source.exists():
        return
    rows = pd.read_csv(source)
    additional = root / f"results/{protocol}/sensitivity/{population}"
    extra_source = additional / source.name
    if extra_source.exists():
        complete_record(additional / status.name)
        extra = pd.read_csv(extra_source)
        rows = pd.concat([rows, extra.loc[extra.condition.str.startswith("mi_matched_")]], ignore_index=True)
    if population == "validation":
        rows = rows.loc[rows.scope.eq("pooled_validation")]
    rows = rows.loc[rows.class_label.eq("normal")]
    if rows.empty:
        return
    keys = ["model", "source_file"]
    if not rows.groupby(keys).full_normal_false_alarm_rate.nunique().eq(1).all():
        raise ValueError("Full source error rates differ across matched comparisons.")
    full = rows.drop_duplicates(keys)[keys+["full_normal_false_alarm_rate"]].rename(
        columns={"full_normal_false_alarm_rate": "rate"}).assign(condition="full")
    reduced = rows[keys+["condition", "reduced_normal_false_alarm_rate"]].rename(
        columns={"reduced_normal_false_alarm_rate": "rate"})
    combined = pd.concat([full, reduced], ignore_index=True)
    order = ["full", "shap_25", "shap_50", "mi_25", "mi_50"]
    names = {"full": "Full", "shap_25": "SHAP 25%", "shap_50": "SHAP 50%", "mi_25": "MI 25%", "mi_50": "MI 50%"}
    if rows.condition.str.startswith("mi_matched_").any():
        order += ["mi_matched_25", "mi_matched_50"]
        names.update(mi_matched_25="Matched MI 25%", mi_matched_50="Matched MI 50%")
    index = pd.MultiIndex.from_product([settings.CLASSIFIER_PANEL, order], names=["model", "condition"])
    values = combined.pivot(index=["model", "condition"], columns="source_file", values="rate").loc[index]
    labels = []
    for source in values.columns:
        stem = Path(source).stem
        label = ("Normal with fault" if "normalfault" in stem else
                 "Normal PTP" if "PTP" in stem else "Main normal")
        labels.append(label)
    fig, ax = plt.subplots(figsize=(10, max(11, .32*len(values)+2)), layout="constrained")
    image = ax.imshow(values, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set(xticks=range(len(values.columns)), xticklabels=labels,
           yticks=range(len(values)), yticklabels=[NAMES[m]+" / "+names[c] for m,c in index],
           title="Normal-source false alarms: " + ("fixed test population" if population == "test" else "pooled held-fold predictions"))
    for i in range(len(values)):
        for j in range(len(values.columns)):
            rate = values.iloc[i,j]
            ax.text(j, i, f"{100*rate:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if rate > .65 else "black")
    fig.colorbar(image, ax=ax, label="False-alarm rate", shrink=.6)
    fig.supxlabel("Rates rounded to 0.1 percentage point; source identities are analysis metadata only.", fontsize=9)
    folder_name = "test" if population == "test" else "reduction"
    save(fig, root / f"figures/{protocol}/{folder_name}", "normal_source_false_alarms")



def test_figures(root, protocol="iec104"):
    classes_config = settings.protocol_settings(protocol).CLASSES
    results = root / f"results/{protocol}/test"
    path = results / "evaluation.json"
    if not path.exists() or json.loads(path.read_text())["status"] != "complete":
        return
    directory = root / f"figures/{protocol}/test"
    points = pd.read_csv(results / "point_metrics.csv")
    order = ["full", "shap_25", "shap_50", "mi_25", "mi_50"]
    labels = {"full":"Full", "shap_25":"SHAP 25%", "shap_50":"SHAP 50%", "mi_25":"MI 25%", "mi_50":"MI 50%"}
    learned = points[points.model.isin(settings.CLASSIFIER_PANEL)].copy()
    benchmark_path = root / f"results/{protocol}/sensitivity/benchmark/prediction_summary.csv"
    if benchmark_path.exists():
        status = validate_benchmark(root, protocol)
        measured = pd.read_csv(benchmark_path).set_index(["model", "condition"])
        learned["pipeline_prediction_seconds"] = [measured.loc[r.model, r.condition]["median_seconds"]
                                                  for r in learned.itertuples()]
    table = learned.pivot(index="model",columns="condition",values="macro_f1").loc[list(settings.CLASSIFIER_PANEL),order]
    table = table.rename(index=NAMES,columns=labels)
    fig,ax = plt.subplots(figsize=(12,5),layout="constrained")
    table.plot.bar(ax=ax,width=.8)
    ax.set(ylabel="Test macro F1",xlabel="",ylim=(0,1.03),title="Reserved-population performance: every declared learned-model condition")
    ax.tick_params(axis="x",rotation=0)
    ax.legend(fontsize=8,ncol=5,loc="lower center")
    save(fig,directory,"macro_f1")

    intervals = pd.read_csv(results/"paired_intervals.csv")
    if len(intervals):
        pairs = intervals[intervals.metric=="macro_f1"].copy()
        fig,ax = plt.subplots(figsize=(10,9),layout="constrained")
        for i,row in enumerate(pairs.itertuples()):
            ax.plot([row.lower,row.upper],[i,i],color="#23778b")
            ax.scatter(row.difference,i,color="#bd5727",s=22)
        text = [NAMES[r.model]+" / "+labels[r.condition] for r in pairs.itertuples()]
        ax.set(yticks=range(len(pairs)),yticklabels=text,xlabel="Reduced minus full macro F1",
               title="Paired test differences: conditional 95% group-bootstrap percentiles")
        ax.axvline(0,color="grey",linestyle="--");ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
        fig.supxlabel("Fixed models and observed measurement groups; intervals do not cover independent captures.",fontsize=9)
        save(fig,directory,"paired_macro_f1")

    else:
        pairs = pd.read_csv(results/"paired_comparisons.csv")
        fig,ax = plt.subplots(figsize=(10,9),layout="constrained")
        for i,row in enumerate(pairs.itertuples()):
            ax.scatter(row.difference_macro_f1,i,color="#bd5727",s=25)
        text = [NAMES[r.model]+" / "+labels[r.condition] for r in pairs.itertuples()]
        ax.set(yticks=range(len(pairs)),yticklabels=text,xlabel="Reduced minus full macro F1",
               title="Paired test differences: point estimates; primary intervals unsupported")
        ax.axvline(0,color="grey",linestyle="--");ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
        fig.supxlabel("PTP test packets occupy one dependency group; no bootstrap intervals were calculated.",fontsize=9)
        save(fig,directory,"paired_macro_f1")

    pairs = pd.read_csv(results/"class_paired_comparisons.csv")
    pairs["row"] = pairs.model.map(NAMES)+" / "+pairs.condition.map(labels)
    classes = pd.read_csv(results/"class_metrics.csv")
    support = classes[(classes.model==settings.CLASSIFIER_PANEL[0])&(classes.condition=="full")].set_index("class_label").support
    fig,axes = plt.subplots(1,2,figsize=(17,10),layout="constrained")
    for ax,metric in zip(axes,("recall","precision")):
        values = pairs.pivot(index="row",columns="class_label",values="difference_"+metric)[list(classes_config)]
        extent = max(.05,float(np.abs(values.to_numpy()).max()))
        im = ax.imshow(values,vmin=-extent,vmax=extent,cmap="RdBu",aspect="auto")
        ax.set(xticks=range(len(classes_config)),xticklabels=[CLASS_NAMES[c]+f" (n={support[c]})" for c in classes_config],
               yticks=range(len(values)),yticklabels=values.index,title="Test "+metric+": reduced minus full")
        plt.setp(ax.get_xticklabels(),rotation=60,ha="right",fontsize=8)
        for i in range(len(values)):
            for j in range(len(classes_config)):
                v = values.iloc[i,j]
                ax.text(j,i,f"{v:+.2f}",ha="center",va="center",fontsize=7,color="white" if abs(v)>.65*extent else "black")
        fig.colorbar(im,ax=ax,shrink=.65)
    save(fig,directory,"class_changes")

    fig,axes = plt.subplots(1,3,figsize=(15,5),layout="constrained")
    for name in settings.CLASSIFIER_PANEL:
        rows = learned[learned.model==name].set_index("condition")
        for ax,column in zip(axes,("development_fit_seconds","pipeline_bytes","pipeline_prediction_seconds")):
            ax.plot(range(4),rows.loc[order[1:],column]/rows.loc["full",column],"o-",label=NAMES[name])
    for ax,title in zip(axes,("Recorded development fitting","Saved pipeline size","Repeated test pipeline prediction")):
        ax.axhline(1,color="grey",linestyle="--")
        ax.set(xticks=range(4),xticklabels=[labels[c] for c in order[1:]],ylabel="Reduced / full",title=title)
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right");ax.grid(axis="y",alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle("Measured costs: descriptive local timings; no live extraction or energy measurement")
    save(fig,directory,"computation_ratios")

    if protocol == "iec61850":
        counts = pd.read_csv(results / "confusion_matrices.csv")
        fig, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained")
        for ax, name in zip(axes.flat, settings.CLASSIFIER_PANEL):
            part = counts[(counts.model == name) & (counts.condition == "full")]
            matrix = part.pivot(index="true_class", columns="predicted_class", values="records").loc[
                list(classes_config), list(classes_config)].to_numpy()
            proportions = matrix / matrix.sum(axis=1, keepdims=True)
            ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues")
            ax.set(xticks=range(len(classes_config)), xticklabels=[CLASS_NAMES[c] for c in classes_config],
                   yticks=range(len(classes_config)), yticklabels=[CLASS_NAMES[c] for c in classes_config],
                   xlabel="Predicted class", ylabel="True class", title=NAMES[name])
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
            for i, j in np.argwhere(matrix > 0):
                ax.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if proportions[i,j] > .6 else "black")
        axes.flat[-1].axis("off")
        fig.suptitle("Full-input test confusions: counts; colour normalised within true class")
        save(fig, directory, "full_confusions")



def sensitivity_figures(root, protocol="iec104"):
    """Plot fold consistency, sampling sensitivity and repeated prediction times."""
    results = root / f"results/{protocol}/sensitivity"
    directory = root / f"figures/{protocol}/sensitivity"
    validate_evaluation(root, protocol)
    record, _ = validate_evaluation(root, protocol, "mi_matched")
    complete_record(results / "ranking_consistency.json")
    status = validate_benchmark(root, protocol)
    agreement = pd.read_csv(results / "rank_agreement.csv")
    labels = {"shap": "Common SHAP", "mi": "MI", "mi_matched": "MI on SHAP rows"}
    agreement["pair"] = agreement.fold_a.str.replace("fold_", "", regex=False) + "–" + agreement.fold_b.str.replace("fold_", "", regex=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    images = {}
    for column, fraction in enumerate(settings.RETENTION_FRACTIONS):
        part = agreement.loc[agreement.retention_fraction.eq(fraction)]
        for row, metric in enumerate(("jaccard", "chance_adjusted_overlap")):
            ax = axes[row, column]
            matrix = part.pivot(index="selector", columns="pair", values=metric).loc[list(labels)]
            images[row] = ax.imshow(matrix, vmin=0 if row == 0 else -1, vmax=1,
                                   cmap="YlGnBu" if row == 0 else "RdBu", aspect="auto")
            ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
                   yticks=range(len(matrix)), yticklabels=[labels[k] for k in matrix.index],
                   xlabel="Training-fold pair", title=f"{int(100*fraction)}% field retention")
            for i in range(len(matrix)):
                for j in range(len(matrix.columns)):
                    v = matrix.iloc[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > .65 else "black")
    for row, label in enumerate(("Jaccard overlap", "Chance-adjusted overlap (Kuncheva)")):
        fig.colorbar(images[row], ax=list(axes[row]), shrink=.8, label=label)
    fig.suptitle("Training-fold selected-set agreement\nOverlapping training populations; descriptive values")
    save(fig, directory, "ranking_consistency")

    base = pd.read_csv(root / f"results/{protocol}/test/point_metrics.csv")
    extra = pd.read_csv(results / "test/point_metrics.csv")
    points = pd.concat([base, extra.loc[extra.condition.str.startswith("mi_matched_")]], ignore_index=True)
    order = [f"{method}_{int(100*f)}" for method in ("shap", "mi", "mi_matched") for f in settings.RETENTION_FRACTIONS]
    short_labels = {"shap": "Common\nSHAP", "mi": "MI", "mi_matched": "Matched\nMI"}
    names = [f"{short_labels[method]}\n{int(100*f)}%" for method in labels for f in settings.RETENTION_FRACTIONS]
    indexed = points.set_index(["model", "condition"])
    rare_class = "mitm" if protocol == "iec104" else "ptp_attack"
    support = pd.read_csv(root / f"results/{protocol}/test/class_metrics.csv")
    support = int(support.loc[support.condition.eq("full") & support.model.eq(settings.CLASSIFIER_PANEL[0])
                              & support.class_label.eq(rare_class), "support"].item())
    groups = record["uncertainty"]["groups_per_class"][rare_class]
    component_word = "component" if groups == 1 else "components"
    rare_title = (f"MITM recall change (n={support})" if protocol == "iec104" else
                  f"PTP recall change ({support} packets in {groups} dependency {component_word})")
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True, layout="constrained")
    for ax, metric, title in zip(axes, ("macro_f1", "normal_false_alarm_rate", "rare_recall"),
                                ("Macro F1 change", "Normal false-alarm change (increase is unfavourable)", rare_title)):
        if metric == "rare_recall":
            cls = pd.concat([pd.read_csv(root / f"results/{protocol}/test/class_metrics.csv"),
                pd.read_csv(results / "test/class_metrics.csv").query("condition.str.startswith('mi_matched_')", engine="python")])
            values = cls.loc[cls.class_label.eq(rare_class)].set_index(["model", "condition"])
            matrix = np.array([[values.loc[name, c]["recall"] - values.loc[name, "full"]["recall"] for c in order]
                               for name in settings.CLASSIFIER_PANEL])
        else:
            matrix = np.array([[indexed.loc[name, c][metric] - indexed.loc[name, "full"][metric] for c in order]
                               for name in settings.CLASSIFIER_PANEL])
        # One colour unit is one percentage point; signs retain the metric meaning.
        matrix *= 100
        extent = max(1., float(np.abs(matrix).max()))
        image = ax.imshow(matrix, vmin=-extent, vmax=extent,
                          cmap="RdBu_r" if metric == "normal_false_alarm_rate" else "RdBu", aspect="auto")
        ax.set(xticks=range(len(order)), xticklabels=names, yticks=range(len(settings.CLASSIFIER_PANEL)),
               yticklabels=[NAMES[n] for n in settings.CLASSIFIER_PANEL], title=title)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(v) > .65*extent else "black")
        fig.colorbar(image, ax=ax, shrink=.85, label="Percentage points")
    protocol_title = "IEC 104" if protocol == "iec104" else "IEC 61850"
    fig.suptitle(protocol_title + ": performance changes after feature reduction\nMatched MI uses SHAP foreground rows; changes are reduced minus full", fontsize=12)
    fig.supxlabel("Cell values rounded to 0.1 percentage point.", fontsize=9)
    save(fig, directory, "sampling_comparison")

    benchmark = results / "benchmark"
    times = pd.read_csv(benchmark / "prediction_summary.csv").set_index(["model", "condition"])
    matrix = np.array([[times.loc[name, c]["reduced_full_median_ratio"] for c in order] for name in settings.CLASSIFIER_PANEL])
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=max(1.25, float(matrix.max())), cmap="YlOrRd", aspect="auto")
    ax.set(xticks=range(len(order)), xticklabels=names, yticks=range(len(settings.CLASSIFIER_PANEL)),
           yticklabels=[NAMES[n] for n in settings.CLASSIFIER_PANEL],
           title=f"Prepared-input prediction: reduced/full median time over {status['repeats']} interleaved rounds",
           xlabel="Matched MI uses the saved SHAP foreground observations")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Time ratio; below 1 means faster")
    save(fig, directory, "prediction_benchmark")


def chapter_figures(root):
    """Format eight thesis figures from saved results, without scoring or fitting.

    Primary conditions stay separate from matched-observation MI. Prediction
    cost panels explicitly identify the supplementary interleaved benchmark.
    """
    from matplotlib.colors import TwoSlopeNorm

    root = Path(root)
    out = root / "figures/comparison"
    models = list(settings.CLASSIFIER_PANEL)
    short = dict(zip(models, ("LR", "DT", "RF", "XGB", "LGB")))
    conditions = ["shap_25", "shap_50", "mi_25", "mi_50"]
    labels = ["SHAP 25%", "SHAP 50%", "MI 25%", "MI 50%"]
    index = pd.MultiIndex.from_product([models, conditions], names=["model", "condition"])
    row_labels = [short[m] + " / " + labels[conditions.index(c)] for m, c in index]
    colours = ["#23778b", "#55a1ac", "#b65323", "#d59668"]
    title = {"iec104": "IEC 104", "iec61850": "IEC 61850"}

    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8}):
        for protocol in title:
            validate_evaluation(root, protocol)
            benchmark_status = validate_benchmark(root, protocol)
            folder = root / f"results/{protocol}/test"
            points = pd.read_csv(folder / "point_metrics.csv").set_index(["model", "condition"])
            pairs = pd.read_csv(folder / "paired_comparisons.csv").set_index(["model", "condition"]).loc[index]
            intervals = pd.read_csv(folder / "paired_intervals.csv")
            fig, ax = plt.subplots(figsize=(6.4, 7.2), layout="constrained")
            for i, ((m, c), row) in enumerate(pairs.iterrows()):
                colour = colours[conditions.index(c)]
                if len(intervals):
                    ci = intervals[(intervals.model == m) & (intervals.condition == c)
                                   & (intervals.metric == "macro_f1")].iloc[0]
                    ax.plot([ci.lower, ci.upper], [i, i], color=colour, linewidth=1.4)
                ax.scatter(row.difference_macro_f1, i, color=colour, s=24, zorder=3)
            ax.set(yticks=range(20), yticklabels=row_labels, xlabel="Macro F1 change: reduced minus full",
                   title=title[protocol] + ": paired test performance")
            ax.invert_yaxis()
            ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
            for boundary in (3.5, 7.5, 11.5, 15.5): ax.axhline(boundary, color="#dddddd", linewidth=.6)
            ax.grid(axis="x", alpha=.2)
            note = ("Points: paired differences; lines: conditional 95% intervals."
                    if len(intervals) else "Point estimates only; primary group intervals were not estimable.")
            fig.supxlabel(note, fontsize=8)
            save(fig, out, f"test_macro_f1_change_{protocol}", dpi=300)

            cp = pd.read_csv(folder / "class_paired_comparisons.csv")
            classes = list(settings.protocol_settings(protocol).CLASSES)
            matrix = cp.pivot(index=["model", "condition"], columns="class_label",
                              values="difference_recall").loc[index, classes] * 100
            cm = pd.read_csv(folder / "class_metrics.csv")
            support = cm[(cm.model == models[0]) & (cm.condition == "full")].set_index("class_label").support
            extent = max(1., float(np.abs(matrix.to_numpy()).max()))
            fig, ax = plt.subplots(figsize=(6.4, 7.3), layout="constrained")
            im = ax.imshow(matrix, cmap="RdBu", vmin=-extent, vmax=extent, aspect="auto")
            names = {"iec104_starvation": "Starvation", "ntp_ddos": "NTP DDoS", "port_scan": "Port scan"}
            ax.set(yticks=range(20), yticklabels=row_labels, xticks=range(len(classes)),
                   xticklabels=[names.get(c, CLASS_NAMES[c])+f"\n(n={support[c]:,})" for c in classes],
                   title=title[protocol] + ": test class-recall changes")
            plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
            for i in range(len(matrix)):
                for j in range(len(classes)):
                    v = matrix.iloc[i, j]
                    label = "0.0" if abs(v) < .05 else f"{v:+.1f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8.2,
                            color="white" if abs(v) > .6*extent else "black")
            for boundary in (3.5, 7.5, 11.5, 15.5): ax.axhline(boundary, color="#777777", linewidth=.6)
            fig.colorbar(im, ax=ax, shrink=.7, label="Recall change (percentage points)", pad=.02)
            save(fig, out, f"test_class_recall_change_{protocol}", dpi=300)

            # Fitting and saved size are primary records; prediction uses the
            # explicitly labelled later benchmark, without changing any model.
            benchmark = pd.read_csv(root / f"results/{protocol}/sensitivity/benchmark/prediction_summary.csv")
            benchmark = benchmark.set_index(["model", "condition"])
            fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.1), layout="constrained")
            for ax, column, panel_title in zip(axes,
                    ("development_fit_seconds", "median_seconds", "pipeline_bytes"),
                    ("Recorded fitting time", f"Prediction: {benchmark_status['repeats']}-round benchmark (exploratory)",
                     "Saved pipeline size")):
                data = benchmark if column == "median_seconds" else points
                values = np.array([[data.loc[(m,c), column]/data.loc[(m,"full"), column]
                                    for c in conditions] for m in models])
                image = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.5), aspect="auto")
                ax.set(yticks=range(5), yticklabels=[short[m] for m in models], xticks=range(4),
                       xticklabels=labels, title=panel_title)
                for i in range(5):
                    for j in range(4):
                        ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=9,
                                color="white" if values[i,j] < .3 or values[i,j] > 2 else "black")
            fig.colorbar(image, ax=list(axes), shrink=.55, label="Reduced / full; below 1 is lower")
            fig.suptitle(title[protocol] + ": recorded computational costs", fontsize=11)
            save(fig, out, f"test_computation_ratios_{protocol}", dpi=300)

        sources = pd.read_csv(root / "results/iec61850/test/source_metrics.csv")
        sources = sources[(sources.class_label == "normal") & sources.model.isin(models)
                          & sources.condition.isin(["full", *conditions])]
        order = pd.MultiIndex.from_product([models, ["full", *conditions]], names=["model", "condition"])
        files = list(sources.source_file.unique())
        files.sort(key=lambda s: 2 if "normalfault" in s else 1 if "PTP" in s else 0)
        matrix = sources.pivot(index=["model", "condition"], columns="source_file",
                               values="normal_false_alarm_rate").loc[order, files] * 100
        fig, ax = plt.subplots(figsize=(6.4, 7.8), layout="constrained")
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="Reds", aspect="auto")
        counts = sources.groupby("source_file").records.first()
        ax.set(yticks=range(25), yticklabels=[short[m]+" / "+("Full" if c=="full" else labels[conditions.index(c)]) for m,c in order],
               xticks=range(3), xticklabels=[name+f"\n(n={counts[f]:,})" for name,f in zip(
                   ["Main normal", "Normal PTP", "Normal with fault"], files)],
               title="IEC 61850: normal-source test false alarms")
        for i in range(25):
            for j in range(3):
                v = matrix.iloc[i,j]
                ax.text(j,i,f"{v:.1f}%",ha="center",va="center",fontsize=8,color="white" if v>60 else "black")
        for boundary in (4.5,9.5,14.5,19.5): ax.axhline(boundary,color="#777777",linewidth=.6)
        fig.colorbar(image, ax=ax, shrink=.65, label="Normal false-alarm rate (%)")
        save(fig, out, "test_normal_source_false_alarms_iec61850", dpi=300)

        fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), layout="constrained")
        for ax, protocol in zip(axes, title):
            rank = pd.read_csv(root / f"results/{protocol}/feature_selection/common_ranking.csv")
            rank = rank[rank.scope == "development"].sort_values("selection_order").head(10)
            display = {"history_frame_len_mean":"Prior frame length: mean", "history_frame_len_std":"Prior frame length: standard deviation",
                       "history_frame_len_max":"Prior frame length: maximum", "history_frame_len_min":"Prior frame length: minimum"}
            ax.barh(range(10), rank.mean_rank, color="#23778b", height=.7)
            ax.set(yticks=range(10), yticklabels=[display.get(s,s) for s in rank.field],
                   xlabel="Mean model rank (smaller is higher-ranked)", title=title[protocol])
            ax.invert_yaxis(); ax.grid(axis="x",alpha=.2)
        fig.suptitle("Leading common SHAP fields: development population", fontsize=11)
        save(fig, out, "common_shap_leading_fields", dpi=300)
    print("Saved eight chapter PNGs from existing results; no fitting or prediction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), default="iec104")
    parser.add_argument("--sensitivity", action="store_true", help="Plot the additional sampling and timing comparisons only.")
    parser.add_argument("--chapter", action="store_true", help="Format eight Results chapter figures from both protocols.")
    args = parser.parse_args()
    if args.chapter:
        chapter_figures(settings.PROJECT_ROOT)
        return
    if args.sensitivity:
        sensitivity_figures(settings.PROJECT_ROOT, args.protocol)
        print("Saved PNG sensitivity figures from numerical tables; no fitting or test scoring.")
        return
    # Reject obsolete evaluations before creating any figures from their tables.
    evaluation = settings.PROJECT_ROOT / f"results/{args.protocol}/test/evaluation.json"
    if evaluation.exists():
        validate_evaluation(settings.PROJECT_ROOT, args.protocol)
    validation_figures(settings.PROJECT_ROOT, args.protocol)
    selection_figures(settings.PROJECT_ROOT, args.protocol)
    reduction_figures(settings.PROJECT_ROOT, args.protocol)
    source_error_figures(settings.PROJECT_ROOT, args.protocol)
    # Test figures require their own completed evaluation.
    test_figures(settings.PROJECT_ROOT, args.protocol)
    source_error_figures(settings.PROJECT_ROOT, args.protocol, "test")
    print("Saved PNG figures from numerical results; no fitting or test scoring.")


if __name__ == "__main__":
    from src.plot_results import main as run_cli
    run_cli()

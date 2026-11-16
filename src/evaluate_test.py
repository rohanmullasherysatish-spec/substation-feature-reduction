"""Evaluate every declared saved final pipeline; no fitting or selection.

IEC104 uncertainty uses the paired whole-group calculation.
IEC61850 reports unsupported primary intervals without splitting mixed groups.
Any intervals concern fixed models and observed groups, not unseen captures.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix

from src import settings
from src.evaluate_validation import confusion_rows, full_record, prediction_table, saved_pipeline, timed_probabilities
from src.modelling import classification_measures
from src.checks import require, invalidate_results, validate_reduction, validate_fitted_parameters
from src.train_full_models import atomic_json, load_training_inputs, protocol_classes
from src.train_reduced_models import reduction_recipe, subset_processor, overlap_count


def plan(protocol="iec104", selector=None):
    """Declare every comparison and uncertainty setting before predicting test rows."""
    require(selector in (None, "mi_matched"), "Unknown additional selector.")
    conditions = (["full", *(f"{selector}_{int(100*f)}" for f in settings.RETENTION_FRACTIONS)]
                  if selector else ["full", "shap_25", "shap_50", "mi_25", "mi_50"])
    value = {"conditions": conditions,
            "models": list(settings.CLASSIFIER_PANEL), "reference": "dummy/full",
            "primary_metric": "macro_f1", "class_metrics": ["precision", "recall", "f1", "errors"],
            "point_comparison": "reduced minus the same classifier's full model; identical observations",
            "retention_verdict": "observed differences; no equivalence test or acceptable-loss threshold",
            "bootstrap": {"replicates": settings.BOOTSTRAP_REPLICATES, "seed": settings.BOOTSTRAP_SEED,
                          "minimum_groups_per_class": settings.MIN_BOOTSTRAP_GROUPS_PER_CLASS,
                          "unit": "saved full-input measurement group", "stratification": "true class",
                          "pairing": "identical sampled group multiplicities for every condition",
                          "interval": "2.5th and 97.5th percentiles of paired differences",
                          "interpretation": "conditional on observed groups, fitted models and representation; not independent-capture uncertainty"},
            "fitting_performed": False,
            "prediction_timing": "three repeated full-pipeline predictions after a 16-row warm-up; descriptive local wall time"}


    if protocol == "iec61850":
        value["bootstrap"].update(
            unit="saved whole block/measurement dependency component",
            stratification="not executed: mixed-label components cannot be divided into pure class strata",
            policy="primary all-class intervals unsupported when a class has fewer than two components; no row bootstrap or surrogate intervals")
        value["source_analysis"] = "all nine source captures; identifiers used only as evaluation metadata"
    if selector:
        value["comparison_role"] = "matched-observation MI sampling sensitivity on the same fixed population"
    protocol_classes(protocol)
    return value


def packet_uncertainty(test):
    """Report support before any resampling; never split mixed dependency groups."""
    labels = protocol_classes("iec61850")
    support = {c: int(test.loc[test.class_label.eq(c), "group_id"].nunique()) for c in labels}
    insufficient = [c for c,n in support.items() if n < settings.MIN_BOOTSTRAP_GROUPS_PER_CLASS]
    require(bool(insufficient), "Packet uncertainty policy requires review if group support changes.")
    details = {"status": "not_estimable", "groups_per_class": support,
        "requested_replicates": settings.BOOTSTRAP_REPLICATES, "executed_replicates": 0,
        "insufficient_classes": insufficient,
        "mixed_label_groups": int(test.groupby("group_id").class_label.nunique().gt(1).sum()),
        "reason": "PTP attack has one test component; primary all-class group intervals lack replicated class support. Mixed components are retained whole; class-pure flow resampling is not applied.",
        "scope": "point estimates only; no formal equivalence or independent-capture inference"}
    return (pd.DataFrame(columns=["model","condition","metric","difference","lower","upper","replicates"]),
            pd.DataFrame(columns=["model","condition","class_label","recall_difference_lower",
                                  "recall_difference_upper","test_groups","replicates"]), details)


def group_error_operator(test, predictions):
    """Represent each weighted confusion as perfect diagonals plus sparse errors."""
    y = pd.Categorical(test.class_label, categories=protocol_classes("iec104")).codes.astype(int)
    inverse, ids = pd.factorize(test.group_id, sort=True)
    table = pd.DataFrame({"group": test.group_id, "target": y}).groupby("group", sort=True).target
    require((table.nunique() == 1).all(), "Class-stratified bootstrap requires class-pure groups.")
    gclass = table.first().to_numpy()
    sizes = np.bincount(inverse, minlength=len(ids))
    rr, cc, vv = [], [], []
    k = len(protocol_classes("iec104"))
    for m, predicted in enumerate(predictions):
        errors = np.flatnonzero(predicted != y)
        rr.extend([inverse[errors], inverse[errors]])
        cc.extend([m*k*k + y[errors]*k + predicted[errors], m*k*k + y[errors]*k + y[errors]])
        vv.extend([np.ones(len(errors), dtype=np.int64), -np.ones(len(errors), dtype=np.int64)])
    operator = coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                          shape=(len(ids), len(predictions)*k*k), dtype=np.int64).tocsr()
    return operator, gclass, sizes


def scores_from_counts(matrix):
    """Calculate fixed-label metrics for all conditions without rebuilding predictions."""
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    support, predicted = matrix.sum(axis=-1), matrix.sum(axis=-2)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros_like(tp), where=(support+predicted)>0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support>0)
    normal = protocol_classes("iec104").index("normal")
    return {"macro_f1": f1.mean(axis=-1), "accuracy": tp.sum(axis=-1)/support.sum(axis=-1),
            "weighted_f1": (f1*support).sum(axis=-1)/support.sum(axis=-1),
            "normal_false_alarm_rate": 1-recall[:, normal]}, recall


def bootstrap(test, predictions, entries, point_metrics):
    """Use one shared set of group draws; never resample individual dependent rows."""
    operator, gclass, sizes = group_error_operator(test, predictions)
    pools = [np.flatnonzero(gclass == c) for c in range(len(protocol_classes("iec104")))]
    support = {label: len(pool) for label, pool in zip(protocol_classes("iec104"), pools)}
    details = {"groups_per_class": support, "requested_replicates": settings.BOOTSTRAP_REPLICATES,
               "executed_replicates": 0, "insufficient_classes": [c for c,n in support.items() if n < settings.MIN_BOOTSTRAP_GROUPS_PER_CLASS]}
    index = {(m,c):i for i,(m,c) in enumerate(entries)}
    paired = [(i,index[m,"full"]) for i,(m,c) in enumerate(entries) if c != "full"]
    if details["insufficient_classes"]:
        details["status"] = "not_estimable"
        return (pd.DataFrame(columns=["model","condition","metric","difference","lower","upper","replicates"]),
                pd.DataFrame(columns=["model","condition","class_label","recall_difference_lower",
                                      "recall_difference_upper","test_groups","replicates"]), details)
    k, m = len(protocol_classes("iec104")), len(entries)
    unit = np.asarray(operator.T @ np.ones(len(sizes),dtype=np.int64)).reshape(m,k,k).astype(float)
    unit[:,np.arange(k),np.arange(k)] += np.bincount(gclass,weights=sizes,minlength=k)
    baseline,_ = scores_from_counts(unit)
    for metric,values in baseline.items():
        require(np.allclose(values,point_metrics[metric],atol=1e-12,rtol=1e-12),
                "Sparse bootstrap operator does not reproduce actual point metrics.")
    draws = {name: np.empty((settings.BOOTSTRAP_REPLICATES,m)) for name in
             ("macro_f1","accuracy","weighted_f1","normal_false_alarm_rate")}
    recalls = np.empty((settings.BOOTSTRAP_REPLICATES,m,k))
    rng = np.random.default_rng(settings.BOOTSTRAP_SEED)
    for repeat in range(settings.BOOTSTRAP_REPLICATES):
        weights = np.zeros(len(sizes), dtype=np.int64)
        for pool in pools:
            weights += np.bincount(rng.choice(pool,len(pool),replace=True), minlength=len(sizes))
        counts = np.asarray(operator.T @ weights).reshape(m,k,k).astype(float)
        counts[:,np.arange(k),np.arange(k)] += np.bincount(gclass,weights=sizes*weights,minlength=k)
        require((counts >= 0).all() and (counts.sum(axis=2)>0).all(), "Invalid resampled confusion counts.")
        values, recall = scores_from_counts(counts)
        for name in draws:
            draws[name][repeat] = values[name]
        recalls[repeat] = recall
    rows, class_rows = [], []
    points = point_metrics.set_index(["model","condition"])
    for i,j in paired:
        model,condition = entries[i]
        for metric,array in draws.items():
            lower,upper = np.quantile(array[:,i]-array[:,j],[.025,.975])
            rows.append({"model":model,"condition":condition,"metric":metric,
                         "difference":points.loc[model,condition][metric]-points.loc[model,"full"][metric],
                         "lower":lower,"upper":upper,"replicates":settings.BOOTSTRAP_REPLICATES})
        for c,label in enumerate(protocol_classes("iec104")):
            lower,upper = np.quantile(recalls[:,i,c]-recalls[:,j,c],[.025,.975])
            class_rows.append({"model":model,"condition":condition,"class_label":label,
                               "recall_difference_lower":lower,"recall_difference_upper":upper,
                               "test_groups":support[label],"replicates":settings.BOOTSTRAP_REPLICATES})
    details.update(status="complete",executed_replicates=settings.BOOTSTRAP_REPLICATES)
    return pd.DataFrame(rows), pd.DataFrame(class_rows), details


def evaluate(root=settings.PROJECT_ROOT, overwrite=False, protocol="iec104", selector=None):
    root = Path(root).resolve()
    labels = protocol_classes(protocol)
    suffix = "sensitivity/" if selector else ""
    result_root = root / f"results/{protocol}/{suffix}"
    results = result_root / "test"
    destination = root / f"data/{protocol}/{suffix}test_predictions.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(results.resolve() == results and destination.resolve() == destination,
            "Test output is redirected.")
    require(overwrite or not results.exists() and not destination.exists(), "Test outputs exist; use --overwrite only for deliberate reproduction.")
    full = full_record(root, protocol)
    reduction, resolved_fields = validate_reduction(root, protocol, selector)
    require(reduction["status"] == "complete" and reduction["recipe"] == reduction_recipe(protocol, selector), "Reduced training is incomplete or stale.")
    if protocol == "iec61850" or selector:
        require(reduction["full_training_recipe"] == full["recipe"]
                and reduction["full_selected_parameters"] == {
                    name: full["selected"][name]["parameters"] for name in settings.CLASSIFIER_PANEL},
                "Reduced training used a different full-model representation or recipe.")
    development, _, _, processors = load_training_inputs(root, protocol)
    processor = processors["development"]
    if protocol == "iec61850":
        from src.iec61850.partition import read_observations
        observations, _ = read_observations(root)
    else:
        from src.iec104.partition import read_observations
        observations = read_observations(root)
    assignments = pd.read_parquet(root / f"data/{protocol}/assignments.parquet")
    identities = ["record_id","dataset_row","source_file","source_row","class_label","group_id"]
    if protocol == "iec104":
        identities.remove("group_id")
    require(observations[identities].equals(assignments[identities]), "Test identities differ from saved allocation.")
    test = observations.loc[assignments.partition.eq("test")].reset_index(drop=True)
    metadata = assignments.loc[assignments.partition.eq("test")].reset_index(drop=True)
    require(not set(metadata.group_id) & set(assignments.loc[assignments.partition.eq("development"), "group_id"]),
            "A dependency group crosses development and test.")
    y = pd.Categorical(test.class_label,categories=labels).codes
    require(set(y) == set(range(len(labels))), "A test class is absent.")
    features = pd.read_csv(result_root / "feature_selection/selected_features.csv")
    fits = pd.read_csv(result_root / "validation/reduced_models.csv")
    full_fits = pd.read_csv(root / f"results/{protocol}/validation/full_models.csv")
    index = {(e["model"],e["condition"]):root/e["path"] for e in reduction["pipelines"] if e["scope"] == "development"}
    conditions = plan(protocol, selector)["conditions"]
    entries = [(name,condition) for name in settings.CLASSIFIER_PANEL for condition in conditions] + [("dummy","full")]
    require(len(entries) == len(settings.CLASSIFIER_PANEL)*len(conditions)+1
            and len(index) == len(settings.CLASSIFIER_PANEL)*(len(conditions)-1), "Final comparison coverage differs.")
    results.mkdir(parents=True,exist_ok=True)
    metrics,classes,confusions,errors,predictions,overlaps = [],[],[],[],[],{}
    source_rows, source_pairs = [], []
    record = {"status":"running","plan":plan(protocol, selector),"test_records":len(test),"test_groups":metadata.group_id.nunique(),"completed_conditions":[]}
    record.update(protocol=protocol, full_training_recipe=full["recipe"],
                  reduced_training_recipe=reduction["recipe"],
                  selected_parameters=reduction["full_selected_parameters"], selected_fields=resolved_fields)
    with TemporaryDirectory(prefix=f".test_{protocol}_",dir=root) as folder, ExitStack() as stack:
        staging = Path(folder)
        invalidate_results(root, protocol, "evaluation", selector)
        atomic_json(results/"evaluation.json",record,staging)
        writer = None
        try:
            for name,condition in entries:
                if condition == "full":
                    pipeline,path = saved_pipeline(root,full,name,"development",processor,protocol)
                    fit = full_fits[(full_fits.model==name)&(full_fits.scope=="development")].iloc[0]
                else:
                    path = index[name,condition]
                    require(path == root/f"models/{protocol}/reduced/{condition}/{name}/development.joblib", "Unexpected reduced path.")
                    pipeline = joblib.load(path)
                    selector_name,percentage = condition.rsplit("_", 1)
                    selected = features[(features.scope=="development")&(features.selector==selector_name)&np.isclose(features.retention_fraction,int(percentage)/100)].field.tolist()
                    expected = subset_processor(processor,selected)
                    fitting_rows = fits[(fits.model==name)&(fits.scope=="development")&(fits.condition==condition)]
                    require(len(fitting_rows) == 1, "Final fitting record is absent or duplicated.")
                    fit = fitting_rows.iloc[0]
                    validate_fitted_parameters(pipeline, full["selected"][name]["parameters"], fit)
                    require(pipeline.model_name==name and pipeline.training_scope=="development"
                            and pipeline.classes_ == labels
                            and pipeline.preprocessor.fields==expected.fields
                            and pipeline.preprocessor.feature_map_==expected.feature_map_
                            and pd.DataFrame(pipeline.preprocessor.field_records_).equals(pd.DataFrame(expected.field_records_)), "Reduced state differs.")
                    if protocol == "iec61850":
                        for attribute in ("states_", "rules", "vocabularies_", "component_screen_", "policy", "settings"):
                            require(getattr(pipeline.preprocessor, attribute) == getattr(expected, attribute),
                                    f"Saved packet reduced preprocessing differs: {attribute}")
                fields = list(pipeline.preprocessor.fields)
                view = "standardised" if name in settings.STANDARDIZE_NUMERIC_FOR else "unscaled"
                if (condition,view) not in overlaps:
                    train_matrix = pipeline.preprocessor.transform_for_model(development[fields],name)
                    test_matrix = pipeline.preprocessor.transform_for_model(test[fields],name)
                    overlaps[condition,view] = {"condition":condition,"numeric_view":view,"test_records":len(test),
                        "matches_development_float64":overlap_count(train_matrix,test_matrix,np.float64,protocol),
                        "matches_development_float32":overlap_count(train_matrix,test_matrix,np.float32,protocol),"rows_removed":0}
                    del train_matrix,test_matrix
                values,timing = timed_probabilities(pipeline,test[fields])
                overall,detail = classification_measures(y,values, class_labels=labels)
                metrics.append({"model":name,"condition":condition,**overall,**timing,
                                "active_original_fields":len(pipeline.preprocessor.active_fields_),"encoded_columns":pipeline.preprocessor.n_features_out_,
                                "pipeline_bytes":path.stat().st_size,"development_fit_seconds":fit.fit_seconds})
                classes.extend({"model":name,"condition":condition,**r} for r in detail)
                confusions.extend(confusion_rows(y,values,name,"test",condition, class_labels=labels))
                table = prediction_table(metadata,values,name,"test",condition, class_labels=labels)
                arrow = pa.Table.from_pandas(table,preserve_index=False)
                if writer is None:
                    writer = stack.enter_context(pq.ParquetWriter(staging/destination.name,arrow.schema))
                writer.write_table(arrow)
                errors.append(table.loc[table.true_class.ne(table.predicted_class),[c for c in table if not c.startswith("probability_")]])
                predictions.append(values.argmax(axis=1))
                if protocol == "iec61850":
                    wrong = values.argmax(axis=1) != y
                    for (source, label), part in metadata.groupby(["source_file", "class_label"], sort=True):
                        mask = part.index.to_numpy()
                        count = int(wrong[mask].sum())
                        source_rows.append({"model": name, "condition": condition, "source_file": source,
                            "class_label": label, "records": len(mask), "groups": part.group_id.nunique(),
                            "errors": count, "error_rate": count/len(mask),
                            "normal_false_alarm_rate": count/len(mask) if label == "normal" else None})
                record["completed_conditions"].append({"model":name,"condition":condition,"pipeline":path.relative_to(root).as_posix()})
                atomic_json(results/"evaluation.json",record,staging)
                print(f"{name}, {condition}: test macro F1={overall['macro_f1']:.6f}; errors={overall['errors']}.",flush=True)
            points = pd.DataFrame(metrics)
            intervals,class_intervals,uncertainty = (packet_uncertainty(metadata) if protocol == "iec61850"
                                                        else bootstrap(metadata,predictions,entries,points))
            record["uncertainty"] = uncertainty
            indexed = points.set_index(["model","condition"])
            paired = []
            class_pairs = []
            class_index = pd.DataFrame(classes).set_index(["model","condition","class_label"])
            for name,condition in entries:
                if condition == "full":continue
                current,reference = indexed.loc[name,condition],indexed.loc[name,"full"]
                i,j = entries.index((name,condition)),entries.index((name,"full"))
                old_correct,new_correct = predictions[j]==y,predictions[i]==y
                row = {"model":name,"condition":condition,"new_errors":int(np.sum(old_correct&~new_correct)),"corrected_errors":int(np.sum(~old_correct&new_correct))}
                for metric in ("macro_f1","accuracy","weighted_f1","normal_false_alarm_rate","errors"):
                    row.update({"full_"+metric:reference[metric],"reduced_"+metric:current[metric],"difference_"+metric:current[metric]-reference[metric]})
                require(row["difference_errors"] == row["new_errors"]-row["corrected_errors"],"Paired test errors disagree.")
                paired.append(row)
                if protocol == "iec61850":
                    for (source, label), part in metadata.groupby(["source_file", "class_label"], sort=True):
                        mask = part.index.to_numpy()
                        before, after = int((~old_correct[mask]).sum()), int((~new_correct[mask]).sum())
                        source_pairs.append({"model": name, "condition": condition, "source_file": source,
                            "class_label": label, "records": len(mask), "full_errors": before, "reduced_errors": after,
                            "difference_errors": after-before,
                            "new_errors": int((old_correct[mask] & ~new_correct[mask]).sum()),
                            "corrected_errors": int((~old_correct[mask] & new_correct[mask]).sum()),
                            "both_wrong": int((~old_correct[mask] & ~new_correct[mask]).sum()),
                            "full_error_rate": before/len(mask), "reduced_error_rate": after/len(mask),
                            "difference_error_rate": (after-before)/len(mask),
                            "full_normal_false_alarm_rate": before/len(mask) if label == "normal" else None,
                            "reduced_normal_false_alarm_rate": after/len(mask) if label == "normal" else None})
                for label in labels:
                    current_class,reference_class = class_index.loc[name,condition,label],class_index.loc[name,"full",label]
                    require(current_class.support==reference_class.support,"Paired test class support differs.")
                    pair = {"model":name,"condition":condition,"class_label":label,"support":int(current_class.support)}
                    for metric in ("precision","recall","f1","errors"):
                        pair.update({"full_"+metric:reference_class[metric],"reduced_"+metric:current_class[metric],
                                     "difference_"+metric:current_class[metric]-reference_class[metric]})
                    class_pairs.append(pair)
            tables = {"point_metrics.csv":points,"class_metrics.csv":pd.DataFrame(classes),"confusion_matrices.csv":pd.DataFrame(confusions),
                      "errors.csv":pd.concat(errors,ignore_index=True),"paired_comparisons.csv":pd.DataFrame(paired),
                      "class_paired_comparisons.csv":pd.DataFrame(class_pairs),
                      "paired_intervals.csv":intervals,"class_recall_intervals.csv":class_intervals,"representation_overlap.csv":pd.DataFrame(overlaps.values())}
            if protocol == "iec61850":
                support = [{"class_label": label, "records": int(metadata.class_label.eq(label).sum()),
                            "groups": int(metadata.loc[metadata.class_label.eq(label), "group_id"].nunique())}
                           for label in labels]
                tables.update({"source_metrics.csv": pd.DataFrame(source_rows),
                               "source_paired_comparisons.csv": pd.DataFrame(source_pairs),
                               "support.csv": pd.DataFrame(support)})
            for filename,table in tables.items():
                table.to_csv(staging/filename,index=False)
                (staging/filename).replace(results/filename)
            writer.close()
            (staging/destination.name).replace(destination)
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(results/"evaluation.json",record,staging)
    if selector:
        from src.selection_checks import selector_comparisons
        selector_comparisons(root, protocol)
    print(f"Evaluated {len(entries)} saved pipelines; uncertainty support recorded; no fitting or selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("iec104", "iec61850"), default="iec104")
    parser.add_argument("--selector", choices=("mi_matched",), help="Evaluate the additional MI conditions and their full references.")
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args()
    evaluate(overwrite=args.overwrite, protocol=args.protocol, selector=args.selector)


if __name__ == "__main__":
    main()

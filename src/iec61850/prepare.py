"""Prepare the declared source-capped packet representation with causal history.

Scan sources to verify widths, labels and counts, or explicitly reuse their
unchanged validated samples. Apply the declared schema and complete-block sample;
this command performs no fitted preprocessing, model fitting or score selection.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.iec61850 import settings as cfg
from src.iec61850.parsing import HISTORY_FIELDS, block_history, numeric_codes, represent, scan_source, sampling_plan
from src.checks import require, invalidate_results


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def dependencies(codes, metadata):
    """Keep whole history blocks and equal full numeric measurements connected."""
    vectors = codes.groupby(list(codes.columns),sort=False,dropna=False).ngroup().to_numpy()
    block_codes,names = pd.factorize(metadata.record_block_id,sort=True)
    parent = np.arange(len(names))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for vector,block in zip(vectors,block_codes):
        if int(vector) in first:
            a,b = find(int(block)),find(first[int(vector)])
            if a != b: parent[max(a,b)] = min(a,b)
        else:
            first[int(vector)] = int(block)
    roots = np.asarray([find(int(i)) for i in block_codes])
    metadata["measurement_group_id"] = vectors
    metadata["group_id"] = pd.factorize(roots,sort=True)[0]
    grouped = metadata.groupby("group_id",sort=True)
    components = grouped.agg(records=("record_id","size"),blocks=("record_block_id","nunique"),
                             classes=("class_label",lambda v:",".join(sorted(set(v))))).reset_index()
    require(metadata.groupby("record_block_id").group_id.nunique().eq(1).all(),"A history block was split.")
    require(metadata.groupby("measurement_group_id").group_id.nunique().eq(1).all(),"Equal numeric measurements cross groups.")
    return components


def prepare(root=cfg.PROJECT_ROOT, overwrite=False, resume=False, reuse_samples=False):
    root = Path(root).resolve()
    results = root/"results/iec61850/preparation"
    output = root/"data/iec61850"
    cache = root/".cache/iec61850_samples"
    status_path = results/"preparation.json"
    require(not(overwrite and resume),"Choose overwrite or resume, not both.")
    require(not (resume and reuse_samples), "Resume and cached-sample replacement are separate operations.")
    prior = json.loads(status_path.read_text()) if status_path.exists() else None
    if reuse_samples:
        require(overwrite and prior is not None and prior["status"] == "complete",
                "Cached-sample reuse requires a completed preparation and --overwrite.")
        current = cfg.preparation_recipe()
        require(all(prior["recipe"][key] == current[key]
                    for key in ("classes", "sampling", "sampling_seed", "sources")),
                "Source inventory or sampling rules changed; a fresh scan is required.")
        require(len(prior["completed_sources"]) == len(cfg.sources()), "Cached scans are incomplete.")
    if resume:
        require(prior is not None and prior["status"] in {"running","failed","interrupted"}
                and prior["recipe"]==cfg.preparation_recipe(),"No compatible interrupted preparation to resume.")
    else:
        require(overwrite or prior is None and not (output/"observations.parquet").exists(),
                "Packet outputs exist; deliberate replacement requires --overwrite.")
    for path in (results,output,cache):
        require(path.resolve()==path,"Output directory is redirected.")
        path.mkdir(parents=True,exist_ok=True)
    source_root = root/"Datasets/iec61850"
    all_fields = [s.strip().rstrip(",") for s in (source_root/"headers_iec61850_all.txt").read_text().splitlines() if s.strip()]
    selected = [s.strip().rstrip(",") for s in (source_root/"headers_iec61850.txt").read_text().splitlines() if s.strip()]
    fields = cfg.field_settings()
    require(len(all_fields)==len(set(all_fields)) and set(selected).issubset(all_fields),"Invalid supplied schema.")
    require([r["field"] for r in fields["field_rules"]]==[f for f in selected if f!="Label"],"Field declarations differ from author-listed candidates.")
    clock = perf_counter()
    record = {"status":"running","recipe":cfg.preparation_recipe(),"completed_sources":prior["completed_sources"] if resume or reuse_samples else [],
              "packages":{name:importlib.metadata.version(name) for name in ("numpy","pandas","pyarrow")},
              "raw_schema_columns":len(all_fields),"author_candidate_inputs":len(selected)-1,"models_fitted":0, "reused_validated_source_samples":bool(reuse_samples or resume),
              "fresh_source_records_read":0}
    invalidate_results(root, "iec61850", "preparation")
    write_json(status_path,record)
    try:
        tables,plans,inventory = [],[],[]
        for index,item in enumerate(cfg.sources()):
            source = source_root/item["path"]
            folder = cache/Path(item["path"]).stem
            folder.mkdir(exist_ok=True)
            old = next((v for v in record["completed_sources"] if v["source_file"]==item["path"]),None)
            if old is not None:
                require(old["source_bytes"]==source.stat().st_size and old["source_mtime_ns"]==source.stat().st_mtime_ns,
                        "A source changed after its cached scan.")
                require((folder/"sampled_strings.parquet").is_file(),"Cached source sample is absent.")
                info = old
                print(f"Reusing the validated sample for {item['path']}.",flush=True)
            else:
                before = source.stat().st_mtime_ns
                print(f"Reading {item['path']} completely; selecting fixed source blocks.",flush=True)
                info = scan_source(source,item,index,all_fields,selected,
                                   {"sampling":cfg.SAMPLING,"seeds":{"sampling":cfg.SAMPLING_SEED}},folder)
                require(source.stat().st_mtime_ns==before,"Source changed during reading.")
                record["fresh_source_records_read"] += info["fresh_validated_records"]
                info["source_mtime_ns"] = before
                record["completed_sources"].append(info)
                write_json(status_path,record)
            require(info["all_record_widths_and_labels_checked"] and info["sample_parquet_roundtrip_exact"],
                    "The cached source lacks completed source validation.")
            table = pq.read_table(folder/"sampled_strings.parquet")
            plan = pd.read_csv(folder/"sampling_blocks.csv")
            expected_plan = sampling_plan(item["records"], index, cfg.SAMPLING, cfg.SAMPLING_SEED)
            pd.testing.assert_frame_equal(plan, pd.DataFrame(expected_plan))
            expected_rows = np.concatenate([np.arange(p["source_row_start_zero_based"],
                                p["source_row_stop_exclusive"]) for p in expected_plan])
            expected_blocks = np.concatenate([np.full(p["records"], p["source_block_index"])
                                              for p in expected_plan])
            require(table.num_rows == info["selected_records"]
                    and np.array_equal(table["source_row"].to_numpy(), expected_rows + 1)
                    and np.array_equal(table["source_block_index"].to_numpy(), expected_blocks),
                    "Cached sample identities differ from the fixed sampling plan.")
            require(set(table["source_file"].to_pylist()) == {item["path"]}
                    and set(table["source_index"].to_pylist()) == {index}
                    and set(table["scenario"].to_pylist()) == {item["scenario"]}
                    and set(table["class_label"].to_pylist()) == {item["class_label"]}
                    and {v.strip() for v in table["Label"].to_pylist()} == {item["raw_label"]},
                    "Cached source identities or labels differ from the declared source.")
            tables.append(table)
            plans.append(plan)
            inventory.append(info)
        raw = pa.concat_tables(tables)
        require(raw.num_rows == sum(item["selected_records"] for item in inventory),
                "Cached sample record count differs from the validated scans.")
        print(f"Parsing {raw.num_rows:,} selected packet observations.",flush=True)
        typed,codes,diagnostics = represent(raw,fields)
        metadata = raw.select(["source_index","source_file","source_row","source_block_index","scenario",
                               "class_label","block_inclusion_probability"]).to_pandas()
        metadata.insert(0,"record_id",metadata.source_file+"::"+metadata.source_row.astype(str))
        metadata.insert(1,"dataset_row",np.arange(len(metadata),dtype=np.int64))
        metadata["record_block_id"] = metadata.source_file+"::block="+metadata.source_block_index.astype(str)
        metadata["raw_label"] = raw["Label"].to_pylist()
        require(metadata.record_id.is_unique and set(metadata.class_label)==set(cfg.CLASSES),"Packet identity/class coverage differs.")
        length = typed["frame.len"].to_numpy(zero_copy_only=False)
        interval = typed["frame.time_delta"].to_numpy(zero_copy_only=False).copy()
        protocol = typed["frame.protocols"].to_pylist()
        history = np.empty((len(metadata),len(HISTORY_FIELDS)))
        starts = np.r_[0,np.flatnonzero(metadata.record_block_id.to_numpy()[1:]!=metadata.record_block_id.to_numpy()[:-1])+1]
        stops = np.r_[starts[1:],len(metadata)]
        context_start = np.empty(len(metadata),dtype=np.int64)
        for start,stop in zip(starts,stops):
            require(np.array_equal(metadata.source_row.iloc[start:stop],np.arange(metadata.source_row.iloc[start],metadata.source_row.iloc[start]+stop-start)),
                    "Selected source block is not contiguous.")
            h = block_history(length[start:stop],interval[start:stop],protocol[start:stop],cfg.PREVIOUS_PACKETS)
            history[start:stop] = h["features"].to_numpy()
            interval[start:stop] = h["adjusted_frame_time_delta"]
            context_start[start:stop] = metadata.source_row.iloc[start]+h["context_start_position"]
        metadata["history_context_start_source_row"] = context_start
        metadata["history_context_stop_source_row_exclusive"] = metadata.source_row
        metadata["block_first_interval_masked"] = False
        metadata.loc[starts,"block_first_interval_masked"] = True
        array = pa.array(interval,from_pandas=True)
        for diagnostic in diagnostics:
            if diagnostic["field"] == "frame.time_delta":
                diagnostic["raw_missing_records"] = diagnostic["missing_records"]
                diagnostic["missing_records"] = array.null_count
                diagnostic["block_boundary_intervals_masked"] = len(starts)
                diagnostic["distinct_values"] = len(pa.compute.dictionary_encode(array).dictionary) + int(array.null_count>0)
        typed = typed.set_column(typed.schema.get_field_index("frame.time_delta"),"frame.time_delta",array)
        codes["frame.time_delta"] = numeric_codes(array)
        feature_rules = [r for r in fields["field_rules"] if r["role"]=="eligible"]
        for index,name in enumerate(HISTORY_FIELDS):
            array = pa.array(history[:,index],type=pa.int64() if name.endswith("_count") else pa.float64(),from_pandas=True)
            typed = typed.append_column(name,array)
            codes[name] = numeric_codes(array)
            feature_rules.append({"field":name,"feature_unit":name,"kind":"int64" if name.endswith("_count") else "float64",
                                  "role":"eligible","mi_component_type":"discrete" if name.endswith("_count") else "continuous",
                                  "numeric_grouping_field":True,"list_component_limit":None,"reason":"Causal preceding-packet summary within a retained source block."})
            diagnostics.append({**feature_rules[-1],"missing_records":array.null_count,
                                "distinct_values":len(pa.compute.dictionary_encode(array).dictionary)+int(array.null_count>0),
                                "maximum_observed_list_items":0,"nonfinite_tokens_to_missing":0})
        print("Connecting whole source blocks and equal numeric measurements at float32 precision.",flush=True)
        components = dependencies(codes,metadata)
        require((metadata.history_context_start_source_row<=metadata.source_row).all(),"A history uses future rows.")
        observations = pa.Table.from_pandas(metadata,preserve_index=False)
        for field in typed.column_names:
            observations = observations.append_column(field,typed[field])
        temporary = output/"observations.parquet.tmp"
        pq.write_table(observations,temporary,compression="zstd",row_group_size=8192)
        require(pq.read_table(temporary).equals(observations),"Prepared packet roundtrip differs.")
        temporary.replace(output/"observations.parquet")
        feature_map = {"classes":list(cfg.CLASSES),"feature_rules":feature_rules,"predictor_fields":[r["field"] for r in feature_rules],
                       "metadata_fields":metadata.columns.tolist(),"numeric_grouping_fields":list(codes.columns),"grouping_dtype":cfg.GROUPING_DTYPE}
        write_json(output/"features.json",feature_map)
        classes = metadata.groupby("class_label").agg(records=("record_id","size"),source_blocks=("record_block_id","nunique"),
                    measurement_groups=("measurement_group_id","nunique"),dependency_groups=("group_id","nunique")).reset_index()
        for filename,table in (("sources.csv",pd.DataFrame(inventory)),("fields.csv",pd.DataFrame(diagnostics)),
                               ("classes.csv",classes),("sampling_blocks.csv",pd.concat(plans,ignore_index=True)),("components.csv",components)):
            table.to_csv(results/filename,index=False)
        record.update(status="complete",parsed_source_records=sum(v["fresh_validated_records"] for v in inventory),
                      prepared_records=len(metadata),prepared_original_fields=len(feature_rules),source_blocks=len(starts),
                      dependency_groups=int(metadata.group_id.nunique()),largest_group=int(components.records.max()),
                      conflicting_label_components=int(components.classes.str.contains(",").sum()),
                      minimum_groups_for_test_and_three_validation_folds_necessary=bool(classes.dependency_groups.min()>=4),
                      actual_partition_feasibility_checked=False,records_removed_or_relabelled=0,elapsed_seconds=perf_counter()-clock)
        print(classes.to_string(index=False),flush=True)
        print(f"Prepared {len(metadata):,} packets and {len(feature_rules)} candidate original units; no classifiers fitted.",flush=True)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(status_path,record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--reuse-samples", action="store_true",
                        help="Reparse unchanged validated cached samples; requires --overwrite.")
    args = parser.parse_args()
    prepare(overwrite=args.overwrite,resume=args.resume,reuse_samples=args.reuse_samples)


if __name__ == "__main__":
    main()

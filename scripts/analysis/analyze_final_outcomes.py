#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("HYPOTHESISMED_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "results/final_outcome_analysis_20260830"
GEN_DIR = ROOT / "results/final_full_benchmark_20260817"
PROTOCOL = ROOT / "results/final_protocol_freeze_20260817"
PARSER_PATH = ROOT / "src/evaluation/parser.py"
FROZEN_PARSER_SHA = "cbaa75e20ff467ed81a3272e79de4ee7e97e3e9e94379ee40ac3d0332c959223"
FROZEN_MEMBERSHIP_SHA = "f335afed6617605d82546c29b51f849339531a2698200d696663145bfe37c8dd"
FROZEN_INPUTS_SHA = "3c75fb55c350f679467b4e2698d1bfb67a22deeb5e49a7b3f4e7caade0a65ebc"
SEED = 20260816
BOOT_B = 2000
BOOT_SEED = 20260830
VALID = {"A", "B", "C", "D", "E"}
DATASETS = ["medqa", "medmcqa", "pubmedqa"]
METHODS = ["direct", "cot", "structured", "fusion"]
GENERATED_METHODS = ["direct", "cot", "structured"]
PRIMARY_FALLBACK_ORDER = ("direct", "cot", "structured")
FALLBACK_ORDERS = list(permutations(GENERATED_METHODS))

MODELS = {
    "qwen36": {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    },
    "gemma4_31b": {
        "model_id": "google/gemma-4-31b-it",
        "revision": "145dc2508c480a64b47242f160d286cff94a2343",
    },
    "medgemma27b": {
        "model_id": "google/medgemma-27b-it",
        "revision": "2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f",
    },
    "qwen25": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "phi4mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def norm_answer(x: Any) -> str:
    if x is None:
        return ""
    y = str(x).strip().upper()
    return y if y in VALID else ""


def verify_frozen_inputs() -> dict[str, str]:
    prompt_parser = PROTOCOL / "final_prompt_parser_hashes.json"
    info = json.loads(prompt_parser.read_text(encoding="utf-8"))
    observed = {
        "parser": sha256_file(PARSER_PATH),
        "standard_membership": sha256_file(ROOT / info["standard_membership"]["path"]),
        "full_generation_inputs": sha256_file(ROOT / info["full_generation_inputs"]["path"]),
        "prompt_parser_manifest": sha256_file(prompt_parser),
    }
    if observed["parser"] != FROZEN_PARSER_SHA:
        raise RuntimeError(f"Parser SHA mismatch: {observed['parser']}")
    if observed["standard_membership"] != FROZEN_MEMBERSHIP_SHA:
        raise RuntimeError(f"Membership SHA mismatch: {observed['standard_membership']}")
    if observed["full_generation_inputs"] != FROZEN_INPUTS_SHA:
        raise RuntimeError(f"Input JSONL SHA mismatch: {observed['full_generation_inputs']}")
    for method in GENERATED_METHODS:
        p = ROOT / info[method]["path"]
        got = sha256_file(p)
        if got != info[method]["sha256"]:
            raise RuntimeError(f"Prompt SHA mismatch for {method}: {got}")
        observed[f"{method}_prompt"] = got
    return observed


def load_parser():
    spec = importlib.util.spec_from_file_location("frozen_hypothesismed_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import parser from {PARSER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_output


def verify_manifests() -> dict[str, dict[str, Any]]:
    manifests = {}
    for model_key, meta in MODELS.items():
        path = GEN_DIR / f"{model_key}_completion_manifest.json"
        if not path.exists():
            raise RuntimeError(f"Missing completion manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "expected_rows": 17868,
            "rows": 17868,
            "unique_keys": 17868,
            "duplicate_keys": 0,
            "complete": True,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise RuntimeError(f"{model_key} manifest {key}={manifest.get(key)!r}, expected {expected!r}")
        if manifest.get("model_id") != meta["model_id"] or manifest.get("revision") != meta["revision"]:
            raise RuntimeError(f"{model_key} manifest model/revision mismatch")
        output = ROOT / manifest["output_jsonl"]
        if sha256_file(output) != manifest["output_sha256"]:
            raise RuntimeError(f"{model_key} output checksum mismatch")
        manifests[model_key] = manifest
    return manifests


def parse_generated_rows(parse_output) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_key, meta in MODELS.items():
        path = GEN_DIR / f"{model_key}_native_chat_generations.jsonl"
        records = load_jsonl(path)
        if len(records) != 17868:
            raise RuntimeError(f"{path} row count mismatch: {len(records)}")
        seen = set()
        for r in records:
            unique_key = tuple(r["unique_key"])
            if unique_key in seen:
                raise RuntimeError(f"Duplicate key in {path}: {unique_key}")
            seen.add(unique_key)
            if r["model_id"] != meta["model_id"] or r["revision"] != meta["revision"]:
                raise RuntimeError(f"Generation model/revision mismatch in {path}:{r['_line_no']}")
            parsed = parse_output(r.get("raw_output") or "")
            answer = norm_answer(parsed.get("answer") if isinstance(parsed, dict) else "")
            confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
            try:
                confidence = float(confidence)
            except Exception:
                confidence = None
            space_label = parsed.get("space_label") if isinstance(parsed, dict) else None
            recovered = bool(answer)
            gold = norm_answer(r.get("gold"))
            method = str(r["method"]).lower()
            exact = r.get("structured_exact_json")
            rows.append(
                {
                    "row_type": "generated",
                    "model_key": model_key,
                    "model_id": r["model_id"],
                    "revision": r["revision"],
                    "backend": r.get("backend"),
                    "runtime_environment": r.get("runtime_environment"),
                    "condition": r.get("condition"),
                    "dataset": r["dataset"],
                    "source_index": int(r["source_index"]),
                    "uid": str(r["uid"]),
                    "gold": gold,
                    "method": method,
                    "seed": int(r.get("seed", SEED)),
                    "controlled_prompt_sha256": r.get("controlled_prompt_sha256"),
                    "serialized_input_sha256": r.get("serialized_input_sha256"),
                    "prompt_tokens": int(r.get("prompt_tokens", 0)),
                    "output_tokens": int(r.get("output_tokens", 0)),
                    "finish_reason": r.get("finish_reason"),
                    "empty_completion": bool(r.get("empty_completion")),
                    "length_truncation": bool(r.get("length_truncation")),
                    "severe_repetition": bool(r.get("severe_repetition")),
                    "reasoning_marker": bool(r.get("reasoning_marker")),
                    "structured_exact_json": bool(exact) if method == "structured" else None,
                    "structured_contract_status": structured_contract_status(method, bool(exact), recovered),
                    "raw_output": r.get("raw_output") or "",
                    "raw_output_path": str(path.relative_to(ROOT)),
                    "raw_output_line": int(r["_line_no"]),
                    "parsed_answer": answer,
                    "parsed_space_label": space_label,
                    "parsed_confidence": confidence,
                    "parser_recovered": recovered,
                    "parser_unrecovered": not recovered,
                    "correct": bool(recovered and answer == gold),
                    "parser_output_json": json.dumps(parsed, sort_keys=True),
                    "fusion_resolution": "",
                    "fusion_fallback_order": "",
                    "fusion_support_count": None,
                    "fusion_supporting_methods": "",
                    "fusion_all_three_agree": None,
                    "fusion_exactly_two_agree": None,
                    "fusion_all_three_disagree": None,
                    "fusion_fallback_used": None,
                    "source_generation_refs": "",
                }
            )
    df = pd.DataFrame(rows)
    expected_n = {"medqa": 1273, "medmcqa": 4183, "pubmedqa": 500}
    count_check = df.groupby(["model_key", "dataset", "method"]).size()
    for model_key in MODELS:
        for dataset, n in expected_n.items():
            for method in GENERATED_METHODS:
                got = int(count_check.get((model_key, dataset, method), 0))
                if got != n:
                    counts = count_check.to_string()
                    raise RuntimeError(f"Unexpected generated method counts:\n{counts}")
    return df


def structured_contract_status(method: str, exact: bool, recovered: bool) -> str:
    if method != "structured":
        return ""
    if exact:
        return "exact_json"
    if recovered:
        return "recoverable_not_exact"
    return "unrecoverable"


def fuse_from_answers(answers: dict[str, str], order: tuple[str, str, str]) -> dict[str, Any]:
    vals = [answers.get(m, "") for m in GENERATED_METHODS if answers.get(m, "") in VALID]
    counts = Counter(vals)
    majority = [a for a, n in counts.items() if n >= 2]
    if majority:
        selected = majority[0]
        resolution = "strict_majority"
    else:
        selected = ""
        resolution = "unresolved"
        for m in order:
            if answers.get(m, "") in VALID:
                selected = answers[m]
                resolution = f"{m}_fallback"
                break
    supporting = [m for m in GENERATED_METHODS if answers.get(m, "") == selected and selected in VALID]
    return {
        "answer": selected,
        "recovered": bool(selected),
        "resolution": resolution,
        "supporting_methods": supporting,
        "support_count": len(supporting),
        "all_three_agree": len(vals) == 3 and len(counts) == 1,
        "exactly_two_agree": sorted(counts.values()) == [1, 2],
        "all_three_disagree": len(vals) == 3 and len(counts) == 3,
        "fallback_used": resolution.endswith("_fallback"),
    }


def add_primary_fusion(df: pd.DataFrame) -> pd.DataFrame:
    generated = df[df["row_type"] == "generated"].copy()
    fusion_rows: list[dict[str, Any]] = []
    group_cols = ["model_key", "model_id", "revision", "condition", "dataset", "source_index", "uid", "gold", "seed"]
    for key, g in generated.groupby(group_cols, sort=False):
        if set(g["method"]) != set(GENERATED_METHODS):
            raise RuntimeError(f"Missing generated method before fusion for {key}: {set(g['method'])}")
        first = g.iloc[0].to_dict()
        by_method = {r.method: r for r in g.itertuples(index=False)}
        answers = {m: str(getattr(by_method[m], "parsed_answer") or "") for m in GENERATED_METHODS}
        fused = fuse_from_answers(answers, PRIMARY_FALLBACK_ORDER)
        refs = [
            {
                "method": m,
                "path": getattr(by_method[m], "raw_output_path"),
                "line": int(getattr(by_method[m], "raw_output_line")),
            }
            for m in GENERATED_METHODS
        ]
        answer = fused["answer"]
        recovered = fused["recovered"]
        gold = key[7]
        fusion_rows.append(
            {
                **{k: first.get(k) for k in df.columns},
                "row_type": "derived_fusion",
                "method": "fusion",
                "prompt_tokens": None,
                "output_tokens": int(sum(int(getattr(by_method[m], "output_tokens")) for m in GENERATED_METHODS)),
                "finish_reason": "derived",
                "empty_completion": False,
                "length_truncation": bool(any(bool(getattr(by_method[m], "length_truncation")) for m in GENERATED_METHODS)),
                "severe_repetition": bool(any(bool(getattr(by_method[m], "severe_repetition")) for m in GENERATED_METHODS)),
                "reasoning_marker": bool(any(bool(getattr(by_method[m], "reasoning_marker")) for m in GENERATED_METHODS)),
                "structured_exact_json": None,
                "structured_contract_status": "",
                "raw_output": "",
                "raw_output_path": "",
                "raw_output_line": None,
                "parsed_answer": answer,
                "parsed_space_label": "",
                "parsed_confidence": None,
                "parser_recovered": recovered,
                "parser_unrecovered": not recovered,
                "correct": bool(recovered and answer == gold),
                "parser_output_json": json.dumps({"fusion_answer": answer}),
                "fusion_resolution": fused["resolution"],
                "fusion_fallback_order": ">".join(PRIMARY_FALLBACK_ORDER),
                "fusion_support_count": fused["support_count"],
                "fusion_supporting_methods": "|".join(fused["supporting_methods"]),
                "fusion_all_three_agree": fused["all_three_agree"],
                "fusion_exactly_two_agree": fused["exactly_two_agree"],
                "fusion_all_three_disagree": fused["all_three_disagree"],
                "fusion_fallback_used": fused["fallback_used"],
                "source_generation_refs": json.dumps(refs, sort_keys=True),
            }
        )
    all_df = pd.concat([df, pd.DataFrame(fusion_rows)], ignore_index=True)
    return all_df


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (max(0.0, center - half), min(1.0, center + half))


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    df.to_csv(path, index=False)
    return path


def summarize_recovery(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = [
        ("overall", []),
        ("model", ["model_key"]),
        ("dataset", ["dataset"]),
        ("method", ["method"]),
        ("model_dataset", ["model_key", "dataset"]),
        ("model_method", ["model_key", "method"]),
        ("dataset_method", ["dataset", "method"]),
        ("model_dataset_method", ["model_key", "dataset", "method"]),
    ]
    for scope, cols in groupings:
        grouped = [((), df)] if not cols else df.groupby(cols, dropna=False, sort=True)
        for key, g in grouped:
            if cols and not isinstance(key, tuple):
                key = (key,)
            data = dict(zip(cols, key if cols else ()))
            n = len(g)
            rec = int(g["parser_recovered"].sum())
            exact_struct = g[g["method"] == "structured"]
            rows.append(
                {
                    "scope": scope,
                    **data,
                    "N": n,
                    "parser_recovery_n": rec,
                    "parser_recovery_rate": rec / n if n else math.nan,
                    "unrecovered_n": n - rec,
                    "unrecovered_rate": (n - rec) / n if n else math.nan,
                    "structured_rows": len(exact_struct),
                    "structured_exact_json_n": int(exact_struct["structured_exact_json"].fillna(False).sum()) if len(exact_struct) else 0,
                    "structured_exact_json_rate": float(exact_struct["structured_exact_json"].fillna(False).mean()) if len(exact_struct) else math.nan,
                    "finish_reason_counts": json.dumps(g["finish_reason"].fillna("").value_counts().to_dict(), sort_keys=True),
                    "length_truncation_n": int(g["length_truncation"].sum()),
                    "length_truncation_rate": float(g["length_truncation"].mean()) if n else math.nan,
                    "severe_repetition_n": int(g["severe_repetition"].sum()),
                    "severe_repetition_rate": float(g["severe_repetition"].mean()) if n else math.nan,
                    "reasoning_marker_n": int(g["reasoning_marker"].sum()),
                    "reasoning_marker_rate": float(g["reasoning_marker"].mean()) if n else math.nan,
                }
            )
    return pd.DataFrame(rows)


def accuracy_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scope, cols in [
        ("model_dataset_method", ["model_key", "dataset", "method"]),
        ("model_method_pooled", ["model_key", "method"]),
        ("dataset_method_pooled", ["dataset", "method"]),
        ("method_pooled", ["method"]),
    ]:
        for key, g in df.groupby(cols, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            d = dict(zip(cols, key))
            n = len(g)
            corr = int(g["correct"].sum())
            rec_n = int(g["parser_recovered"].sum())
            cond_corr = int(g.loc[g["parser_recovered"], "correct"].sum())
            lo, hi = wilson_ci(corr, n)
            clo, chi = wilson_ci(cond_corr, rec_n)
            rows.append(
                {
                    "scope": scope,
                    **d,
                    "N": n,
                    "correct_n": corr,
                    "accuracy_end_to_end": corr / n if n else math.nan,
                    "accuracy_end_to_end_CI95_low": lo,
                    "accuracy_end_to_end_CI95_high": hi,
                    "recovered_N": rec_n,
                    "conditional_correct_n": cond_corr,
                    "accuracy_conditional_recovered": cond_corr / rec_n if rec_n else math.nan,
                    "accuracy_conditional_CI95_low": clo,
                    "accuracy_conditional_CI95_high": chi,
                }
            )
    acc = pd.DataFrame(rows)
    macro_rows = []
    mdm = acc[acc["scope"] == "model_dataset_method"].copy()
    for (model_key, method), g in mdm.groupby(["model_key", "method"], sort=True):
        if set(g["dataset"]) != set(DATASETS):
            continue
        macro_rows.append(
            {
                "model_key": model_key,
                "method": method,
                "datasets": ",".join(DATASETS),
                "equal_weight_macro_accuracy_end_to_end": float(g.set_index("dataset").loc[DATASETS, "accuracy_end_to_end"].mean()),
                "equal_weight_macro_accuracy_conditional": float(g.set_index("dataset").loc[DATASETS, "accuracy_conditional_recovered"].mean()),
                "equal_weight_macro_recovery": float(
                    df[(df["model_key"] == model_key) & (df["method"] == method)]
                    .groupby("dataset")["parser_recovered"].mean()
                    .loc[DATASETS]
                    .mean()
                ),
                "pooled_N": int(g["N"].sum()),
                "pooled_correct_n": int(g["correct_n"].sum()),
            }
        )
    return acc, pd.DataFrame(macro_rows)


def wide_item_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["model_key", "dataset", "source_index", "uid", "gold"]
    for key, g in df.groupby(cols, sort=False):
        out = dict(zip(cols, key))
        for r in g.itertuples(index=False):
            m = r.method
            out[f"{m}_answer"] = r.parsed_answer
            out[f"{m}_recovered"] = bool(r.parser_recovered)
            out[f"{m}_correct"] = bool(r.correct)
            out[f"{m}_structured_exact_json"] = bool(r.structured_exact_json) if m == "structured" else None
        rows.append(out)
    return pd.DataFrame(rows)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, B: int = BOOT_B) -> tuple[float, float]:
    n = len(a)
    if n == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(BOOT_SEED + n)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        diffs[i] = float(np.mean(a[idx] - b[idx]))
    return (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def paired_comparisons(wide: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("cot", "fusion"),
        ("fusion", "cot"),
        ("direct", "cot"),
        ("structured", "cot"),
        ("fusion", "direct"),
    ]
    rows = []
    for model_key, mg in wide.groupby("model_key", sort=True):
        for dataset, dg in mg.groupby("dataset", sort=True):
            rows.extend(paired_rows_for_group(model_key, dataset, dg, pairs))
        # Equal-weight macro bootstrap by sampling within each dataset.
        for a, b in pairs:
            ds_diffs = []
            for dataset in DATASETS:
                dg = mg[mg["dataset"] == dataset]
                ds_diffs.append(float(dg[f"{a}_correct"].astype(float).mean() - dg[f"{b}_correct"].astype(float).mean()))
            obs = float(np.mean(ds_diffs))
            lo, hi = bootstrap_macro_pair(mg, a, b)
            rows.append(
                {
                    "scope": "equal_weight_macro",
                    "model_key": model_key,
                    "dataset": "MACRO",
                    "method_a": a,
                    "method_b": b,
                    "contrast": f"{a}_minus_{b}",
                    "N": int(len(mg)),
                    "accuracy_a": float(np.mean([mg[mg["dataset"] == d][f"{a}_correct"].astype(float).mean() for d in DATASETS])),
                    "accuracy_b": float(np.mean([mg[mg["dataset"] == d][f"{b}_correct"].astype(float).mean() for d in DATASETS])),
                    "accuracy_diff": obs,
                    "CI95_low": lo,
                    "CI95_high": hi,
                    "A_correct_B_wrong": int(((mg[f"{a}_correct"]) & (~mg[f"{b}_correct"])).sum()),
                    "A_wrong_B_correct": int(((~mg[f"{a}_correct"]) & (mg[f"{b}_correct"])).sum()),
                    "discordant_N": int((mg[f"{a}_correct"] != mg[f"{b}_correct"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_rows_for_group(model_key: str, dataset: str, g: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for a, b in pairs:
        av = g[f"{a}_correct"].astype(bool).to_numpy()
        bv = g[f"{b}_correct"].astype(bool).to_numpy()
        diff = av.astype(float) - bv.astype(float)
        lo, hi = bootstrap_diff(av.astype(float), bv.astype(float))
        rows.append(
            {
                "scope": "dataset",
                "model_key": model_key,
                "dataset": dataset,
                "method_a": a,
                "method_b": b,
                "contrast": f"{a}_minus_{b}",
                "N": int(len(g)),
                "accuracy_a": float(av.mean()) if len(av) else math.nan,
                "accuracy_b": float(bv.mean()) if len(bv) else math.nan,
                "accuracy_diff": float(diff.mean()) if len(diff) else math.nan,
                "CI95_low": lo,
                "CI95_high": hi,
                "A_correct_B_wrong": int((av & ~bv).sum()),
                "A_wrong_B_correct": int((~av & bv).sum()),
                "discordant_N": int((av != bv).sum()),
            }
        )
    return rows


def bootstrap_macro_pair(mg: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED + len(mg) + sum(ord(c) for c in a + b))
    diffs = np.empty(BOOT_B)
    by_ds = {d: mg[mg["dataset"] == d] for d in DATASETS}
    for i in range(BOOT_B):
        vals = []
        for d in DATASETS:
            dg = by_ds[d]
            idx = rng.integers(0, len(dg), len(dg))
            av = dg[f"{a}_correct"].astype(float).to_numpy()[idx]
            bv = dg[f"{b}_correct"].astype(float).to_numpy()[idx]
            vals.append(float(np.mean(av - bv)))
        diffs[i] = float(np.mean(vals))
    return (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def common_recovered(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    joint_rows = []
    pairs = [("direct", "cot"), ("structured", "cot"), ("cot", "fusion"), ("fusion", "direct")]
    for model_key, mg in wide.groupby("model_key", sort=True):
        for dataset, dg in mg.groupby("dataset", sort=True):
            for a, b in pairs:
                mask = dg[f"{a}_recovered"].astype(bool) & dg[f"{b}_recovered"].astype(bool)
                sub = dg[mask]
                pair_rows.append(common_pair_record("dataset", model_key, dataset, a, b, len(dg), sub))
            joint_rows.extend(joint_records("dataset", model_key, dataset, dg))
        for a, b in pairs:
            macro_vals = []
            macro_frac = []
            for dataset in DATASETS:
                dg = mg[mg["dataset"] == dataset]
                mask = dg[f"{a}_recovered"].astype(bool) & dg[f"{b}_recovered"].astype(bool)
                sub = dg[mask]
                macro_vals.append(
                    {
                        "acc_a": float(sub[f"{a}_correct"].mean()) if len(sub) else math.nan,
                        "acc_b": float(sub[f"{b}_correct"].mean()) if len(sub) else math.nan,
                    }
                )
                macro_frac.append(len(sub) / len(dg) if len(dg) else math.nan)
            pair_rows.append(
                {
                    "scope": "equal_weight_macro",
                    "model_key": model_key,
                    "dataset": "MACRO",
                    "method_a": a,
                    "method_b": b,
                    "N_all": int(len(mg)),
                    "N_common_recovered": int(((mg[f"{a}_recovered"].astype(bool)) & (mg[f"{b}_recovered"].astype(bool))).sum()),
                    "common_recovered_fraction": float(np.nanmean(macro_frac)),
                    "accuracy_a_common": float(np.nanmean([x["acc_a"] for x in macro_vals])),
                    "accuracy_b_common": float(np.nanmean([x["acc_b"] for x in macro_vals])),
                    "accuracy_diff_common": float(np.nanmean([x["acc_a"] - x["acc_b"] for x in macro_vals])),
                }
            )
        joint_rows.extend(joint_records("equal_weight_macro", model_key, "MACRO", mg))
    return pd.DataFrame(pair_rows), pd.DataFrame(joint_rows)


def common_pair_record(scope: str, model_key: str, dataset: str, a: str, b: str, n_all: int, sub: pd.DataFrame) -> dict[str, Any]:
    return {
        "scope": scope,
        "model_key": model_key,
        "dataset": dataset,
        "method_a": a,
        "method_b": b,
        "N_all": int(n_all),
        "N_common_recovered": int(len(sub)),
        "common_recovered_fraction": len(sub) / n_all if n_all else math.nan,
        "accuracy_a_common": float(sub[f"{a}_correct"].mean()) if len(sub) else math.nan,
        "accuracy_b_common": float(sub[f"{b}_correct"].mean()) if len(sub) else math.nan,
        "accuracy_diff_common": float(sub[f"{a}_correct"].mean() - sub[f"{b}_correct"].mean()) if len(sub) else math.nan,
    }


def joint_records(scope: str, model_key: str, dataset: str, g: pd.DataFrame) -> list[dict[str, Any]]:
    if scope == "equal_weight_macro":
        rows = []
        for denominator, mask_col in [
            ("joint_four_method_recovered", None),
            ("joint_generated_recovered_and_structured_exact_json", "structured_exact"),
        ]:
            vals = []
            fracs = []
            n_joint_total = 0
            for ds in DATASETS:
                dg = g[g["dataset"] == ds]
                mask = (
                    dg["direct_recovered"].astype(bool)
                    & dg["cot_recovered"].astype(bool)
                    & dg["structured_recovered"].astype(bool)
                    & dg["fusion_recovered"].astype(bool)
                )
                if mask_col == "structured_exact":
                    mask = mask & dg["structured_structured_exact_json"].fillna(False).astype(bool)
                sub = dg[mask]
                n_joint_total += len(sub)
                fracs.append(len(sub) / len(dg) if len(dg) else math.nan)
                vals.append({m: float(sub[f"{m}_correct"].mean()) if len(sub) else math.nan for m in METHODS})
            row = {
                "scope": scope,
                "model_key": model_key,
                "dataset": dataset,
                "denominator": denominator,
                "N_all": int(len(g)),
                "N_joint_recovered": int(n_joint_total),
                "joint_recovered_fraction": float(np.nanmean(fracs)),
            }
            for m in METHODS:
                row[f"{m}_accuracy_joint"] = float(np.nanmean([x[m] for x in vals]))
            rows.append(row)
        return rows
    mask = (
        g["direct_recovered"].astype(bool)
        & g["cot_recovered"].astype(bool)
        & g["structured_recovered"].astype(bool)
        & g["fusion_recovered"].astype(bool)
    )
    exact_mask = mask & g["structured_structured_exact_json"].fillna(False).astype(bool)
    rows = []
    for denominator, sub in [
        ("joint_four_method_recovered", g[mask]),
        ("joint_generated_recovered_and_structured_exact_json", g[exact_mask]),
    ]:
        row = {
            "scope": scope,
            "model_key": model_key,
            "dataset": dataset,
            "denominator": denominator,
            "N_all": int(len(g)),
            "N_joint_recovered": int(len(sub)),
            "joint_recovered_fraction": len(sub) / len(g) if len(g) else math.nan,
        }
        for m in METHODS:
            row[f"{m}_accuracy_joint"] = float(sub[f"{m}_correct"].mean()) if len(sub) else math.nan
        rows.append(row)
    return rows


def structured_contract(df: pd.DataFrame) -> pd.DataFrame:
    s = df[(df["method"] == "structured") & (df["row_type"] == "generated")].copy()
    rows = []
    for scope, cols in [
        ("model", ["model_key"]),
        ("model_dataset", ["model_key", "dataset"]),
        ("dataset", ["dataset"]),
        ("overall", []),
    ]:
        grouped = [((), s)] if not cols else s.groupby(cols, sort=True)
        for key, g in grouped:
            if cols and not isinstance(key, tuple):
                key = (key,)
            base = {"scope": scope, **dict(zip(cols, key if cols else ()))}
            n = len(g)
            exact = g["structured_exact_json"].fillna(False).astype(bool)
            recovered = g["parser_recovered"].astype(bool)
            rows.append(
                {
                    "row_kind": "contract_summary",
                    **base,
                    "feature": "",
                    "feature_value": "",
                    "N": n,
                    "exact_json_n": int(exact.sum()),
                    "exact_json_rate": float(exact.mean()) if n else math.nan,
                    "recoverable_but_not_exact_n": int((recovered & ~exact).sum()),
                    "recoverable_but_not_exact_rate": float((recovered & ~exact).mean()) if n else math.nan,
                    "unrecoverable_n": int((~recovered).sum()),
                    "unrecoverable_rate": float((~recovered).mean()) if n else math.nan,
                    "conditional_accuracy_recovered": float(g.loc[recovered, "correct"].mean()) if recovered.any() else math.nan,
                    "end_to_end_accuracy": float(g["correct"].mean()) if n else math.nan,
                    "truncation_rate": float(g["length_truncation"].mean()) if n else math.nan,
                    "reasoning_marker_rate": float(g["reasoning_marker"].mean()) if n else math.nan,
                }
            )
    for feature in ["length_truncation", "reasoning_marker", "severe_repetition"]:
        for (model_key, val), g in s.groupby(["model_key", feature], sort=True):
            exact = g["structured_exact_json"].fillna(False).astype(bool)
            recovered = g["parser_recovered"].astype(bool)
            rows.append(
                {
                    "row_kind": "feature_association",
                    "scope": "model",
                    "model_key": model_key,
                    "dataset": "",
                    "feature": feature,
                    "feature_value": bool(val),
                    "N": len(g),
                    "exact_json_n": int(exact.sum()),
                    "exact_json_rate": float(exact.mean()) if len(g) else math.nan,
                    "recoverable_but_not_exact_n": int((recovered & ~exact).sum()),
                    "recoverable_but_not_exact_rate": float((recovered & ~exact).mean()) if len(g) else math.nan,
                    "unrecoverable_n": int((~recovered).sum()),
                    "unrecoverable_rate": float((~recovered).mean()) if len(g) else math.nan,
                    "conditional_accuracy_recovered": float(g.loc[recovered, "correct"].mean()) if recovered.any() else math.nan,
                    "end_to_end_accuracy": float(g["correct"].mean()) if len(g) else math.nan,
                    "truncation_rate": float(g["length_truncation"].mean()) if len(g) else math.nan,
                    "reasoning_marker_rate": float(g["reasoning_marker"].mean()) if len(g) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def fallback_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    generated = df[df["row_type"] == "generated"].copy()
    base_cols = ["model_key", "dataset", "source_index", "uid", "gold"]
    rows = []
    item_rows = []
    grouped = generated.groupby(base_cols, sort=False)
    for order in FALLBACK_ORDERS:
        order_name = ">".join(order)
        per_order = []
        for key, g in grouped:
            by_method = {r.method: r for r in g.itertuples(index=False)}
            answers = {m: str(getattr(by_method[m], "parsed_answer") or "") for m in GENERATED_METHODS}
            fused = fuse_from_answers(answers, order)
            record = dict(zip(base_cols, key))
            correct = bool(fused["recovered"] and fused["answer"] == record["gold"])
            record.update(
                {
                    "fallback_order": order_name,
                    "fusion_answer": fused["answer"],
                    "fusion_recovered": fused["recovered"],
                    "fusion_correct": correct,
                    "resolution": fused["resolution"],
                    "strict_majority": fused["resolution"] == "strict_majority",
                    "fallback_used": fused["fallback_used"],
                    "cot_correct": bool(getattr(by_method["cot"], "correct")),
                }
            )
            per_order.append(record)
            item_rows.append(record)
        odf = pd.DataFrame(per_order)
        for (model_key, dataset), g in odf.groupby(["model_key", "dataset"], sort=True):
            rows.append(fallback_summary_record("dataset", order_name, model_key, dataset, g))
        for model_key, mg in odf.groupby("model_key", sort=True):
            ds_acc = []
            ds_cot = []
            ds_cov = []
            for d in DATASETS:
                dg = mg[mg["dataset"] == d]
                ds_acc.append(float(dg["fusion_correct"].mean()))
                ds_cot.append(float(dg["cot_correct"].mean()))
                ds_cov.append(float(dg["fusion_recovered"].mean()))
            rec = fallback_summary_record("equal_weight_macro", order_name, model_key, "MACRO", mg)
            rec["fusion_accuracy"] = float(np.mean(ds_acc))
            rec["cot_accuracy"] = float(np.mean(ds_cot))
            rec["fusion_coverage"] = float(np.mean(ds_cov))
            rec["cot_minus_fusion"] = rec["cot_accuracy"] - rec["fusion_accuracy"]
            rows.append(rec)
    item_df = pd.DataFrame(item_rows)
    save_csv(item_df, "fallback_order_sensitivity_item_level.csv")
    return pd.DataFrame(rows)


def fallback_summary_record(scope: str, order_name: str, model_key: str, dataset: str, g: pd.DataFrame) -> dict[str, Any]:
    return {
        "scope": scope,
        "fallback_order": order_name,
        "model_key": model_key,
        "dataset": dataset,
        "N": int(len(g)),
        "cot_accuracy": float(g["cot_correct"].mean()) if len(g) else math.nan,
        "fusion_accuracy": float(g["fusion_correct"].mean()) if len(g) else math.nan,
        "fusion_coverage": float(g["fusion_recovered"].mean()) if len(g) else math.nan,
        "cot_minus_fusion": float(g["cot_correct"].mean() - g["fusion_correct"].mean()) if len(g) else math.nan,
        "strict_majority_rate": float(g["strict_majority"].mean()) if len(g) else math.nan,
        "fallback_rate": float(g["fallback_used"].mean()) if len(g) else math.nan,
    }


def generation_behavior(df: pd.DataFrame) -> pd.DataFrame:
    generated = df[df["row_type"] == "generated"].copy()
    rows = []
    for scope, cols in [
        ("model_method", ["model_key", "method"]),
        ("model_dataset_method", ["model_key", "dataset", "method"]),
        ("model", ["model_key"]),
        ("overall", []),
    ]:
        grouped = [((), generated)] if not cols else generated.groupby(cols, sort=True)
        for key, g in grouped:
            if cols and not isinstance(key, tuple):
                key = (key,)
            row = {"scope": scope, **dict(zip(cols, key if cols else ()))}
            row.update(
                {
                    "N": int(len(g)),
                    "finish_reason_counts": json.dumps(g["finish_reason"].fillna("").value_counts().to_dict(), sort_keys=True),
                    "empty_completion_n": int(g["empty_completion"].sum()),
                    "length_truncation_n": int(g["length_truncation"].sum()),
                    "severe_repetition_n": int(g["severe_repetition"].sum()),
                    "reasoning_marker_n": int(g["reasoning_marker"].sum()),
                    "output_tokens_mean": float(g["output_tokens"].mean()),
                    "output_tokens_median": float(g["output_tokens"].median()),
                    "output_tokens_p95": float(g["output_tokens"].quantile(0.95)),
                    "output_tokens_max": int(g["output_tokens"].max()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def serialization_forensic_summary() -> pd.DataFrame:
    artifacts = [
        ROOT / "results/round4_qwen36_serialization_boundary_20260816/qwen36_serialization_boundary.jsonl",
        ROOT / "results/round4_serialization_boundary_ablation_20260816/qwen_serialization_ablation.jsonl",
        ROOT / "results/round4_serialization_boundary_ablation_20260816/medgemma_serialization_ablation_v3.jsonl",
        ROOT / "results/final_round3_closure_20260814_234647/claude_round2_final_closure_20260816/current_environment_primary_raw_vs_chat.csv",
        ROOT / "results/final_round3_closure_20260814_234647/targeted_robustness_final/serving_output_diagnostics.csv",
    ]
    rows = []
    for path in artifacts:
        if not path.exists():
            rows.append({"artifact": str(path.relative_to(ROOT)), "exists": False})
            continue
        if path.suffix == ".jsonl":
            records = load_jsonl(path)
            df = pd.DataFrame(records)
            variant_col = "variant" if "variant" in df.columns else "condition"
            group_cols = [c for c in ["model", "model_id", variant_col, "method", "dataset"] if c in df.columns]
            for key, g in df.groupby(group_cols, dropna=False, sort=True):
                if not isinstance(key, tuple):
                    key = (key,)
                row = {
                    "artifact": str(path.relative_to(ROOT)),
                    "exists": True,
                    "sha256": sha256_file(path),
                    "rows_in_artifact": len(df),
                    **dict(zip(group_cols, key)),
                    "N": len(g),
                    "finish_reason_counts": json.dumps(g.get("finish_reason", pd.Series(dtype=str)).fillna("").value_counts().to_dict(), sort_keys=True),
                    "length_finish_n": int((g.get("finish_reason", pd.Series(dtype=str)) == "length").sum()) if "finish_reason" in g else 0,
                    "severe_repetition_n": int(g.get("severe_repetition", pd.Series(dtype=bool)).fillna(False).sum()) if "severe_repetition" in g else 0,
                    "reasoning_marker_n": int(g.get("special_reasoning_marker", g.get("reasoning_marker", pd.Series(dtype=bool))).fillna(False).sum()) if ("special_reasoning_marker" in g or "reasoning_marker" in g) else 0,
                    "output_tokens_mean": float(g.get("output_tokens", pd.Series(dtype=float)).mean()) if "output_tokens" in g else math.nan,
                    "note": "Bounded serialization diagnostic; not part of the five-model native_chat benchmark.",
                }
                rows.append(row)
        else:
            df = pd.read_csv(path)
            row = {
                "artifact": str(path.relative_to(ROOT)),
                "exists": True,
                "sha256": sha256_file(path),
                "rows_in_artifact": len(df),
                "N": len(df),
                "columns": ",".join(df.columns.astype(str)),
                "note": "Frozen forensic/serving summary artifact; summarized separately from native_chat benchmark.",
            }
            for col in df.columns:
                low = str(col).lower()
                if "accuracy" in low or "macro" in low or "fusion" in low or "cot_minus" in low:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if vals.notna().any():
                        row[f"{col}_mean"] = float(vals.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(
    checks: dict[str, str],
    manifests: dict[str, dict[str, Any]],
    acc: pd.DataFrame,
    macro: pd.DataFrame,
    paired: pd.DataFrame,
    common: pd.DataFrame,
    joint: pd.DataFrame,
    structured: pd.DataFrame,
    fallback: pd.DataFrame,
    behavior: pd.DataFrame,
) -> None:
    macro_pivot = macro.pivot(index="model_key", columns="method", values="equal_weight_macro_accuracy_end_to_end")
    best_rows = []
    for model, row in macro_pivot.iterrows():
        best_method = row.astype(float).idxmax()
        best_rows.append((model, best_method, float(row[best_method])))
    fusion_vs_cot = paired[
        (paired["scope"] == "equal_weight_macro")
        & (paired["method_a"] == "fusion")
        & (paired["method_b"] == "cot")
    ][["model_key", "accuracy_diff", "CI95_low", "CI95_high"]]
    structured_model = structured[
        (structured["row_kind"] == "contract_summary") & (structured["scope"] == "model")
    ][["model_key", "N", "exact_json_n", "exact_json_rate", "recoverable_but_not_exact_n", "unrecoverable_n", "end_to_end_accuracy", "conditional_accuracy_recovered"]]
    med_behavior = behavior[(behavior["scope"] == "model") & (behavior["model_key"] == "medgemma27b")]

    lines = [
        "# Final Outcome Analysis - 2026-08-30",
        "",
        "Outcome analysis was run only after the frozen full-generation benchmark completed. No additional model outputs were generated, no prompts/parser/fallback rules were modified, and raw generation JSONLs were not overwritten.",
        "",
        "## Frozen Inputs Verified",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## Completion Manifests",
        "",
    ]
    for model, m in manifests.items():
        lines.append(f"- `{model}`: rows={m['rows']}, unique_keys={m['unique_keys']}, duplicate_keys={m['duplicate_keys']}, complete={m['complete']}")
    lines += [
        "",
        "## Primary Macro Accuracy",
        "",
        macro[["model_key", "method", "equal_weight_macro_accuracy_end_to_end", "equal_weight_macro_accuracy_conditional", "equal_weight_macro_recovery"]].to_markdown(index=False),
        "",
        "## Best Method By Model",
        "",
    ]
    for model, method, value in best_rows:
        lines.append(f"- `{model}`: best equal-weight end-to-end macro = `{method}` ({value:.4f})")
    lines += [
        "",
        "## Fusion Versus CoT",
        "",
        fusion_vs_cot.to_markdown(index=False),
        "",
        "Positive values mean Fusion outperformed CoT; negative values mean Fusion underperformed CoT. The result varies by model rather than supporting a universal Fusion benefit.",
        "",
        "## Structured Contract",
        "",
        structured_model.to_markdown(index=False),
        "",
        "Structured exact-JSON compliance is a major observed behavior difference. It should not be interpreted as lack of semantic knowledge by itself; the table separates exact JSON, recoverable-but-not-exact, and unrecovered outputs.",
        "",
        "## Generation Behavior",
        "",
        behavior[behavior["scope"] == "model"][["model_key", "N", "finish_reason_counts", "length_truncation_n", "severe_repetition_n", "reasoning_marker_n", "output_tokens_mean", "output_tokens_p95"]].to_markdown(index=False),
        "",
        "MedGemma reasoning traces were retained and analyzed as native output behavior.",
        "",
        "## Scientific Framing",
        "",
        "- The full five-model benchmark is a native-chat method comparison, not a two-interface experiment.",
        "- Serving/interface sensitivity should be described using the separately frozen serialization-boundary diagnostics.",
        "- The strongest supported finding is model-dependent method behavior under valid native instruction serialization, with large cross-model differences in Structured exact-contract adherence.",
        "- Fusion does not uniformly help; its effect varies by model and dataset and should be reported with paired denominators.",
        "- Recovery-controlled tables distinguish parser/recovery effects from answer correctness on common recoverable items.",
    ]
    (OUT / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    methods = [
        "# Methods Audit",
        "",
        "MODEL_SELECTION_USED_ACCURACY = NO",
        "PANEL_FROZEN_BEFORE_FULL_OUTCOME_ANALYSIS = YES",
        "PROTOCOL_FROZEN_BEFORE_FULL_OUTCOME_ANALYSIS = YES",
        "FULL_PANEL_PRIMARY_CONDITION = native_chat",
        "FULL_PANEL_IS_TWO_INTERFACE_COMPARISON = NO",
        "",
        "Parser: `src/evaluation/parser.py`; SHA256 verified before use.",
        "",
        "Primary Fusion rule: strict majority across Direct, CoT, Structured. If no majority exists, choose the first parser-recovered valid answer in the frozen fallback order Direct, then CoT, then Structured. If none recover, Fusion is unrecovered.",
        "",
        "Six fallback-order sensitivity enumerates all permutations of Direct, CoT, and Structured after strict majority.",
        "",
        "End-to-end accuracy counts unrecovered outputs as incorrect. Conditional accuracy is computed only among parser-recovered outputs. Equal-weight macro accuracy averages MedQA, MedMCQA, and PubMedQA dataset accuracies.",
        "",
        "Paired comparisons preserve item pairing and use bootstrap confidence intervals for accuracy differences.",
    ]
    (OUT / "METHODS_AUDIT.md").write_text("\n".join(methods) + "\n", encoding="utf-8")

    limitations = [
        "# Limitations",
        "",
        "- Backends and decoding settings differ by model family as frozen in the protocol, so cross-model differences should not be attributed solely to architecture.",
        "- Structured exact-JSON compliance differs sharply across models; non-exact outputs can still contain recoverable answers.",
        "- MedGemma produced native reasoning markers in many rows; these traces were retained rather than suppressed.",
        "- Length truncation and severe repetition are analyzed as observed generation behavior, not repaired post hoc.",
        "- The full panel uses only `native_chat`; malformed/bare continuation evidence belongs to the bounded forensic diagnostics.",
        "- Bootstrap CIs are empirical summaries over the frozen item set and do not remove dataset or benchmark limitations.",
    ]
    (OUT / "LIMITATIONS.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")


def checksums() -> None:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.name == "checksums.sha256" or path.is_dir():
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(ROOT)}")
    (OUT / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = verify_frozen_inputs()
    manifests = verify_manifests()
    parse_output = load_parser()
    generated = parse_generated_rows(parse_output)
    item = add_primary_fusion(generated)

    # Keep raw outputs in parquet. The CSV omits raw_output but preserves path/line references to remain manageable.
    try:
        item.to_parquet(OUT / "analysis_ready_item_level.parquet", index=False)
    except Exception as exc:
        (OUT / "analysis_ready_item_level.parquet.ERROR.txt").write_text(str(exc) + "\n", encoding="utf-8")
    csv_item = item.drop(columns=["raw_output"])
    save_csv(csv_item, "analysis_ready_item_level.csv")

    recovery = summarize_recovery(item)
    acc, macro = accuracy_summaries(item)
    wide = wide_item_table(item)
    paired = paired_comparisons(wide)
    common, joint = common_recovered(wide)
    structured = structured_contract(item)
    fallback = fallback_sensitivity(item)
    behavior = generation_behavior(item)
    forensic = serialization_forensic_summary()

    save_csv(recovery, "parser_recovery_summary.csv")
    save_csv(acc, "primary_accuracy_results.csv")
    save_csv(macro, "macro_accuracy_results.csv")
    save_csv(paired, "paired_method_comparisons.csv")
    save_csv(common, "common_recovered_results.csv")
    save_csv(joint, "joint_recovered_results.csv")
    save_csv(structured, "structured_contract_analysis.csv")
    save_csv(fallback, "fallback_order_sensitivity.csv")
    save_csv(behavior, "generation_behavior_summary.csv")
    save_csv(forensic, "serialization_forensic_summary.csv")

    audit = {
        "created": "2026-08-30",
        "script": str((OUT / "analyze_final_outcomes.py").relative_to(ROOT)),
        "script_sha256": sha256_file(OUT / "analyze_final_outcomes.py"),
        "parser_sha256": checks["parser"],
        "standard_membership_sha256": checks["standard_membership"],
        "full_generation_inputs_sha256": checks["full_generation_inputs"],
        "primary_condition": "native_chat",
        "primary_fallback_order": ">".join(PRIMARY_FALLBACK_ORDER),
        "fallback_orders": [">".join(o) for o in FALLBACK_ORDERS],
        "new_model_generations": False,
        "raw_generation_files_modified": False,
        "accuracy_authorized_after_generation_complete": True,
        "models": manifests,
        "row_counts": {
            "generated_rows": int(len(generated)),
            "analysis_ready_rows_including_fusion": int(len(item)),
        },
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(checks, manifests, acc, macro, paired, common, joint, structured, fallback, behavior)
    checksums()
    print(json.dumps({"status": "complete", "out_dir": str(OUT), "rows": len(item)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

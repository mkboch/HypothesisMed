#!/usr/bin/env python3
import json, re, math, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "expanded_fix_noapi"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["medqa", "medmcqa", "pubmedqa"]
SCALED_MODELS = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B",
    "microsoft_phi_4_mini_instruct": "Phi-4-mini",
}
METHODS = ["direct", "cot", "hypothesismed_v3", "fusion"]


def read_jsonl(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def norm_answer(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    m = re.search(r"\b([A-H])\b", s.upper())
    if m:
        return m.group(1)
    y = s.lower()
    if y in {"yes", "no", "maybe"}:
        return y
    return s


def norm_space(x):
    if x is None:
        return ""
    s = str(x).strip().upper()
    if "CONTRADICT" in s or "DUPLICATE" in s or "AMBIG" in s or "NONUNIQUE" in s:
        return "CONTRADICTED"
    if "INCOMPLETE" in s or "MISSING" in s or "REMOVE" in s:
        return "INCOMPLETE"
    if "VALID" in s or "ORIGINAL" in s:
        return "VALID"
    return ""


def to_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def correct(pred, gold):
    p = norm_answer(pred)
    g = norm_answer(gold)
    return bool(p and g and p == g)


def parse_options(x):
    if isinstance(x, dict):
        return {str(k).upper()[:1]: str(v) for k, v in x.items()}
    if isinstance(x, list):
        return {chr(65 + i): str(v) for i, v in enumerate(x)}
    try:
        return parse_options(json.loads(str(x)))
    except Exception:
        return {}


def exact_or_near_duplicate(options):
    opts = parse_options(options)
    vals = []
    for v in opts.values():
        t = re.sub(r"\W+", " ", str(v).lower()).strip()
        if t:
            vals.append(t)
    if len(vals) != len(set(vals)):
        return True
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a, b = vals[i], vals[j]
            if min(len(a), len(b)) >= 10 and (a in b or b in a):
                return True
    return False


def stable_split(model, dataset, qid):
    h = hashlib.md5(f"{model}|{dataset}|{qid}".encode()).hexdigest()
    return "val" if int(h[:8], 16) % 5 == 0 else "test"


def load_scaled_wide():
    rows = []

    for model_slug, model_label in SCALED_MODELS.items():
        for dataset in DATASETS:
            paths = {
                "direct": ROOT / "results" / f"{model_slug}_direct_{dataset}_main_large.jsonl",
                "cot": ROOT / "results" / f"{model_slug}_cot_{dataset}_main_large.jsonl",
                "hypothesismed_v3": ROOT / "results" / f"{model_slug}_hypothesismed_v3_{dataset}_main_large.jsonl",
                "fusion": ROOT / "results" / "fusion" / f"{model_slug}_fusion_majority_answer_hypmed_v3_space_{dataset}_main_large.jsonl",
            }

            method_maps = {}
            for method, path in paths.items():
                rr = read_jsonl(path)
                if not rr:
                    print(f"WARNING missing or empty: {path}")
                    continue
                d = {}
                for r in rr:
                    qid = r.get("id") or r.get("example_id") or r.get("question_id")
                    if qid is not None:
                        d[str(qid)] = r
                method_maps[method] = d

            all_ids = sorted(set().union(*[set(v.keys()) for v in method_maps.values()])) if method_maps else []

            for qid in all_ids:
                base = None
                for m in METHODS:
                    if qid in method_maps.get(m, {}):
                        base = method_maps[m][qid]
                        break
                if base is None:
                    continue

                row = {
                    "model": model_label,
                    "model_slug": model_slug,
                    "dataset": dataset,
                    "id": qid,
                    "question": base.get("question", ""),
                    "options": base.get("options", {}),
                    "gold_answer": norm_answer(base.get("gold_answer", base.get("answer", ""))),
                    "gold_space": norm_space(base.get("gold_space_label", base.get("space_label", "VALID"))),
                    "split": stable_split(model_label, dataset, qid),
                }

                for m in METHODS:
                    r = method_maps.get(m, {}).get(qid, {})
                    row[f"{m}_answer"] = norm_answer(r.get("pred_answer", r.get("answer", "")))
                    row[f"{m}_space"] = norm_space(r.get("pred_space_label", r.get("space_label", "")))
                    row[f"{m}_confidence"] = to_float(r.get("confidence", np.nan))
                    row[f"{m}_parse"] = bool(row[f"{m}_answer"])
                    row[f"{m}_correct"] = correct(row[f"{m}_answer"], row["gold_answer"])

                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No scaled rows found. Check main_large result files.")
    return df


def vote_prediction(row, methods, fallback, weights=None, use_conf=False):
    weights = weights or {m: 1.0 for m in methods}
    scores = defaultdict(float)
    votes = []

    for m in methods:
        a = norm_answer(row.get(f"{m}_answer", ""))
        if not a:
            continue
        w = float(weights.get(m, 1.0))
        if use_conf:
            c = row.get(f"{m}_confidence", np.nan)
            w *= max(0.0, min(1.0, float(c))) if np.isfinite(c) else 0.5
        scores[a] += w
        votes.append(a)

    if not votes:
        return ""

    counts = Counter(votes)
    top_answer, top_count = counts.most_common(1)[0]

    if top_count > len(votes) / 2:
        return top_answer

    max_score = max(scores.values())
    best = [a for a, s in scores.items() if s == max_score]

    if len(best) == 1:
        return best[0]

    for m in fallback:
        a = norm_answer(row.get(f"{m}_answer", ""))
        if a in best:
            return a

    return top_answer


def make_fusion_variants(df):
    df = df.copy()

    df["pred_direct"] = df["direct_answer"]
    df["pred_cot"] = df["cot_answer"]
    df["pred_hypothesismed_v3"] = df["hypothesismed_v3_answer"]
    df["pred_existing_fusion"] = df["fusion_answer"]

    fallback_orders = {
        "majority_DCH": ["direct", "cot", "hypothesismed_v3"],
        "majority_DHC": ["direct", "hypothesismed_v3", "cot"],
        "majority_CDH": ["cot", "direct", "hypothesismed_v3"],
        "majority_CHD": ["cot", "hypothesismed_v3", "direct"],
        "majority_HDC": ["hypothesismed_v3", "direct", "cot"],
        "majority_HCD": ["hypothesismed_v3", "cot", "direct"],
    }

    for name, fallback in fallback_orders.items():
        df[f"pred_{name}"] = df.apply(
            lambda r: vote_prediction(r, ["direct", "cot", "hypothesismed_v3"], fallback),
            axis=1,
        )

    pair_defs = {
        "direct_cot": ["direct", "cot"],
        "direct_hypmed": ["direct", "hypothesismed_v3"],
        "cot_hypmed": ["cot", "hypothesismed_v3"],
    }

    for name, methods in pair_defs.items():
        df[f"pred_{name}"] = df.apply(
            lambda r: vote_prediction(r, methods, methods),
            axis=1,
        )

    df["pred_conf_weighted"] = ""
    df["pred_valacc_weighted"] = ""
    df["pred_parseacc_weighted"] = ""

    for (model, dataset), g in df.groupby(["model", "dataset"]):
        idx = g.index
        val = g[g["split"] == "val"]
        if val.empty:
            val = g

        weights_acc = {}
        weights_parseacc = {}

        for m in ["direct", "cot", "hypothesismed_v3"]:
            acc = float(val[f"{m}_correct"].mean())
            parse = float(val[f"{m}_parse"].mean())
            weights_acc[m] = acc
            weights_parseacc[m] = acc * parse

        df.loc[idx, "pred_conf_weighted"] = g.apply(
            lambda r: vote_prediction(
                r,
                ["direct", "cot", "hypothesismed_v3"],
                ["direct", "cot", "hypothesismed_v3"],
                weights={m: 1.0 for m in ["direct", "cot", "hypothesismed_v3"]},
                use_conf=True,
            ),
            axis=1,
        )

        df.loc[idx, "pred_valacc_weighted"] = g.apply(
            lambda r: vote_prediction(
                r,
                ["direct", "cot", "hypothesismed_v3"],
                ["direct", "cot", "hypothesismed_v3"],
                weights=weights_acc,
            ),
            axis=1,
        )

        df.loc[idx, "pred_parseacc_weighted"] = g.apply(
            lambda r: vote_prediction(
                r,
                ["direct", "cot", "hypothesismed_v3"],
                ["direct", "cot", "hypothesismed_v3"],
                weights=weights_parseacc,
            ),
            axis=1,
        )

    def oracle_any(r):
        for m in ["direct", "cot", "hypothesismed_v3"]:
            if correct(r.get(f"{m}_answer", ""), r["gold_answer"]):
                return r["gold_answer"]
        return vote_prediction(r, ["direct", "cot", "hypothesismed_v3"], ["direct", "cot", "hypothesismed_v3"])

    df["pred_oracle_any_prompt"] = df.apply(oracle_any, axis=1)
    return df


def summarize_variants(df):
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    rows = []

    for scope, d0 in [("test", df[df["split"] == "test"]), ("all", df)]:
        groupings = [
            (["model"], "ALL"),
            (["model", "dataset"], None),
        ]

        for keys, fixed_dataset in groupings:
            for gvals, g in d0.groupby(keys):
                if not isinstance(gvals, tuple):
                    gvals = (gvals,)
                model = gvals[0]
                dataset = fixed_dataset if fixed_dataset else gvals[1]

                for col in pred_cols:
                    corr = [correct(p, gold) for p, gold in zip(g[col], g["gold_answer"])]
                    parse = [bool(norm_answer(p)) for p in g[col]]
                    rows.append({
                        "scope": scope,
                        "model": model,
                        "dataset": dataset,
                        "variant": col.replace("pred_", ""),
                        "N": len(g),
                        "accuracy": float(np.mean(corr)) if len(g) else np.nan,
                        "parse_coverage": float(np.mean(parse)) if len(g) else np.nan,
                        "wrong_parsed_rate": float(np.mean([p and not c for p, c in zip(parse, corr)])) if len(g) else np.nan,
                    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_fusion_strategy_and_prompt_ablation.csv", index=False)
    return out


def add_uncertainty_scores(df):
    df = df.copy()
    df["final_pred"] = df["pred_majority_DCH"]
    df["final_correct"] = [correct(p, g) for p, g in zip(df["final_pred"], df["gold_answer"])]

    vote_agreement = []
    vote_entropy_conf = []

    for _, r in df.iterrows():
        votes = [
            norm_answer(r.get(f"{m}_answer", ""))
            for m in ["direct", "cot", "hypothesismed_v3"]
            if norm_answer(r.get(f"{m}_answer", ""))
        ]

        if not votes:
            vote_agreement.append(0.0)
            vote_entropy_conf.append(0.0)
            continue

        counts = Counter(votes)
        agree = counts.most_common(1)[0][1] / len(votes)
        probs = np.array(list(counts.values()), dtype=float) / len(votes)
        ent = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_ent = math.log(len(counts)) if len(counts) > 1 else 1.0
        ent_conf = 1.0 - min(1.0, ent / max_ent)

        vote_agreement.append(float(agree))
        vote_entropy_conf.append(float(ent_conf))

    df["score_vote_agreement"] = vote_agreement
    df["score_vote_entropy_conf"] = vote_entropy_conf
    df["score_hypmed_conf"] = df["hypothesismed_v3_confidence"].fillna(0.5).clip(0, 1)
    df["score_fusion_conf"] = df["fusion_confidence"].fillna(df["score_hypmed_conf"]).clip(0, 1)
    df["score_space_valid"] = (df["hypothesismed_v3_space"].map(norm_space) == "VALID").astype(float)
    df["score_space_conf"] = df["score_space_valid"] * df["score_hypmed_conf"]
    df["score_hybrid"] = (
        0.40 * df["score_vote_agreement"]
        + 0.30 * df["score_hypmed_conf"]
        + 0.30 * df["score_space_valid"]
    )
    return df


def auroc_binary(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(set(y.tolist())) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc_binary(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    return float(np.trapz(precision, recall))


def risk_coverage_auc(corrects, scores):
    corrects = np.asarray(corrects).astype(bool)
    scores = np.asarray(scores).astype(float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    order = np.argsort(-scores)
    risks = []
    covs = []
    for k in range(1, len(order) + 1):
        chosen = order[:k]
        risk = 1.0 - corrects[chosen].mean()
        cov = k / len(order)
        risks.append(risk)
        covs.append(cov)
    return float(np.trapz(risks, covs))


def summarize_selective_prediction(df):
    score_cols = [
        "score_vote_agreement",
        "score_vote_entropy_conf",
        "score_hypmed_conf",
        "score_fusion_conf",
        "score_space_valid",
        "score_space_conf",
        "score_hybrid",
    ]

    rows = []

    for (model, dataset), g in df.groupby(["model", "dataset"]):
        corrects = g["final_correct"].astype(bool).to_numpy()
        wrongs = (~g["final_correct"].astype(bool)).astype(int).to_numpy()

        for score_col in score_cols:
            scores = g[score_col].astype(float).fillna(0.0).to_numpy()
            order = np.argsort(-scores)
            aurc = risk_coverage_auc(corrects, scores)
            auroc = auroc_binary(wrongs, 1.0 - scores)
            auprc = auprc_binary(wrongs, 1.0 - scores)

            for coverage in [0.80, 0.90, 0.95, 1.00]:
                k = max(1, int(math.ceil(len(g) * coverage)))
                chosen = order[:k]
                acc = float(corrects[chosen].mean())
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "score": score_col.replace("score_", ""),
                    "coverage": coverage,
                    "accepted_N": k,
                    "accepted_accuracy": acc,
                    "risk": 1.0 - acc,
                    "risk_coverage_auc": aurc,
                    "wrong_detection_AUROC": auroc,
                    "wrong_detection_AUPRC": auprc,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_uncertainty_selective_prediction.csv", index=False)
    return out


def summarize_space_triage(df):
    rows = []

    for (model, dataset), g in df.groupby(["model", "dataset"]):
        base_wrong = 1.0 - float(g["final_correct"].mean())
        total_wrong = int((~g["final_correct"]).sum())

        for threshold in [0.30, 0.50, 0.70]:
            conf = g["hypothesismed_v3_confidence"].fillna(0.0)
            space = g["hypothesismed_v3_space"].map(norm_space)
            flag = (space != "VALID") | (conf < threshold)
            accepted = ~flag

            flagged_wrong = int((flag & (~g["final_correct"])).sum())
            flagged_total = int(flag.sum())

            rows.append({
                "model": model,
                "dataset": dataset,
                "threshold": threshold,
                "N": len(g),
                "flag_rate": float(flag.mean()),
                "accepted_N": int(accepted.sum()),
                "accepted_accuracy": float(g.loc[accepted, "final_correct"].mean()) if accepted.sum() else np.nan,
                "flagged_wrong_rate": float((~g.loc[flag, "final_correct"]).mean()) if flagged_total else np.nan,
                "overall_wrong_rate": base_wrong,
                "error_enrichment": float(((~g.loc[flag, "final_correct"]).mean() / base_wrong)) if flagged_total and base_wrong > 0 else np.nan,
                "wrong_answer_recall": float(flagged_wrong / total_wrong) if total_wrong else np.nan,
                "flag_precision_for_wrong": float(flagged_wrong / flagged_total) if flagged_total else np.nan,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_space_guided_triage.csv", index=False)
    return out


def summarize_space_hybrid_existing():
    files = [
        ROOT / "results" / "expanded" / "qwen2_5_7b_instruct_space_stress_outputs.jsonl",
        ROOT / "results" / "expanded" / "microsoft_phi_4_mini_instruct_space_stress_outputs.jsonl",
    ]

    rows = []

    for path in files:
        rr = read_jsonl(path)
        if not rr:
            print(f"WARNING no stress outputs: {path}")
            continue

        for r in rr:
            model_raw = str(r.get("model", path.name)).lower()
            if "qwen" in model_raw or "qwen" in path.name.lower():
                model = "Qwen2.5-7B"
            elif "phi" in model_raw or "phi" in path.name.lower():
                model = "Phi-4-mini"
            else:
                model = r.get("model", path.stem)

            gold = norm_space(r.get("gold_space_label", r.get("gold_space", "")))
            pred = norm_space(r.get("pred_space_label", r.get("space_prediction", "")))
            if not pred:
                pred = norm_space(r.get("pred_space", ""))

            det = "CONTRADICTED" if exact_or_near_duplicate(r.get("options", {})) else ""
            hybrid = det if det else pred

            if not gold:
                continue

            rows.append({
                "source_file": str(path.relative_to(ROOT)),
                "model": model,
                "dataset": r.get("dataset", ""),
                "stress_type": r.get("transform", r.get("stress_type", "")),
                "gold_space": gold,
                "v3_space": pred,
                "det_duplicate_space": det,
                "hybrid_space": hybrid,
                "v3_correct": pred == gold,
                "hybrid_correct": hybrid == gold,
                "hybrid_changed": bool(det and det != pred),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("WARNING no usable SPACE stress-test rows found")
        return df

    df.to_csv(OUT / "space_hybrid_item_level.csv", index=False)

    overall = df.groupby("model", dropna=False).agg(
        N=("gold_space", "size"),
        v3_space_accuracy=("v3_correct", "mean"),
        hybrid_space_accuracy=("hybrid_correct", "mean"),
        changed_rate=("hybrid_changed", "mean"),
    ).reset_index()
    overall.to_csv(OUT / "table_space_v3_vs_deterministic_hybrid_overall.csv", index=False)

    by_label = df.groupby(["model", "gold_space"], dropna=False).agg(
        N=("gold_space", "size"),
        v3_space_accuracy=("v3_correct", "mean"),
        hybrid_space_accuracy=("hybrid_correct", "mean"),
        changed_rate=("hybrid_changed", "mean"),
    ).reset_index()
    by_label.to_csv(OUT / "table_space_v3_vs_deterministic_hybrid_by_label.csv", index=False)

    return df


def summarize_existing_stronger_models():
    patterns = [
        "results/qwen3_30b_thinking_*.jsonl",
        "results/reparsed/qwen3_30b_thinking_*.jsonl",
        "results/*llama*70b*.jsonl",
        "results/*meditron*70b*.jsonl",
        "results/*qwen2_5_72b*.jsonl",
        "results/*openbiollm*.jsonl",
        "results/*med42*.jsonl",
    ]

    files = []
    for pat in patterns:
        files.extend(ROOT.glob(pat))
    files = sorted(set(files))

    rows = []

    for path in files:
        rr = read_jsonl(path)
        if not rr:
            continue

        corr = []
        parse = []
        space_cov = []
        model = None
        method = None
        dataset = None

        for r in rr:
            model = model or r.get("model", path.stem)
            method = method or r.get("method", "")
            dataset = dataset or r.get("dataset", "")
            pred = norm_answer(r.get("pred_answer", r.get("answer", "")))
            gold = norm_answer(r.get("gold_answer", ""))
            corr.append(correct(pred, gold))
            parse.append(bool(pred))
            space_cov.append(bool(norm_space(r.get("pred_space_label", r.get("space_label", "")))))

        rows.append({
            "file": str(path.relative_to(ROOT)),
            "model": model,
            "dataset": dataset,
            "method": method,
            "N": len(rr),
            "accuracy": float(np.mean(corr)) if corr else np.nan,
            "parse_coverage": float(np.mean(parse)) if parse else np.nan,
            "space_coverage": float(np.mean(space_cov)) if space_cov else np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_existing_stronger_model_outputs_inventory.csv", index=False)
    return out


def write_summary(base, fusion_table, selective_table, triage_table, stronger_table):
    path = OUT / "EXPANDED_NOAPI_SUMMARY.md"

    with path.open("w", encoding="utf-8") as f:
        f.write("# Reviewer-2 no-API experiment summary\n\n")
        f.write(f"Output folder: `{OUT}`\n\n")
        f.write(f"Scaled examples loaded: {len(base)}\n\n")

        f.write("## Generated files\n\n")
        for name in [
            "scaled_wide_base_outputs.csv",
            "scaled_wide_with_expanded_variants.csv",
            "table_fusion_strategy_and_prompt_ablation.csv",
            "table_uncertainty_selective_prediction.csv",
            "table_space_guided_triage.csv",
            "table_space_v3_vs_deterministic_hybrid_overall.csv",
            "table_space_v3_vs_deterministic_hybrid_by_label.csv",
            "table_existing_stronger_model_outputs_inventory.csv",
        ]:
            f.write(f"- `{name}`\n")

        f.write("\n## Fusion strategy sample\n\n")
        f.write(fusion_table.head(25).to_string(index=False))

        f.write("\n\n## Selective prediction sample\n\n")
        f.write(selective_table.head(25).to_string(index=False))

        f.write("\n\n## SPACE triage sample\n\n")
        f.write(triage_table.head(25).to_string(index=False))

        if isinstance(stronger_table, pd.DataFrame) and not stronger_table.empty:
            f.write("\n\n## Existing stronger-model outputs found\n\n")
            f.write(stronger_table.head(40).to_string(index=False))

    return path


def main():
    print("STEP 1: loading scaled existing outputs")
    base = load_scaled_wide()
    base.to_csv(OUT / "scaled_wide_base_outputs.csv", index=False)
    print(f"LOADED_ROWS={len(base)}")

    print("STEP 2: creating fusion and ablation variants")
    fused = make_fusion_variants(base)
    fused = add_uncertainty_scores(fused)
    fused.to_csv(OUT / "scaled_wide_with_expanded_variants.csv", index=False)

    print("STEP 3: summarizing fusion strategies and prompt ablations")
    fusion_table = summarize_variants(fused)

    print("STEP 4: selective prediction and uncertainty baselines")
    selective_table = summarize_selective_prediction(fused)

    print("STEP 5: SPACE-guided downstream triage")
    triage_table = summarize_space_triage(fused)

    print("STEP 6: deterministic SPACE hybrid on existing stress outputs")
    summarize_space_hybrid_existing()

    print("STEP 7: inventory existing stronger-model outputs")
    stronger_table = summarize_existing_stronger_models()

    summary = write_summary(base, fusion_table, selective_table, triage_table, stronger_table)

    print(f"NOAPI_OK")
    print(f"OUT={OUT}")
    print(f"SUMMARY={summary}")


if __name__ == "__main__":
    main()

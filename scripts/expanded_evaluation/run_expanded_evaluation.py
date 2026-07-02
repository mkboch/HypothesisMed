#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/home/manikm/HypothesisMed")
OUT = ROOT / "results/expanded"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "medqa": ROOT / "datasets/transformed/medqa_original1000.jsonl",
    "medmcqa": ROOT / "datasets/transformed/medmcqa_original1000.jsonl",
    "pubmedqa": ROOT / "datasets/transformed/pubmedqa_original1000.jsonl",
}

PRIMARY_MODELS = [
    ("qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("microsoft_phi_4_mini_instruct", "microsoft/Phi-4-mini-instruct"),
]

ALL_MODELS = [
    ("qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("microsoft_phi_4_mini_instruct", "microsoft/Phi-4-mini-instruct"),
    ("deepseek_r1_qwen_32b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    ("biomistral_biomistral_7b", "BioMistral/BioMistral-7B"),
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_first(d, keys, default=None):
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return default


def get_question(x):
    q = pick_first(x, ["question", "Question", "query", "input", "stem", "prompt"], "")
    if isinstance(q, list):
        q = " ".join(map(str, q))
    return str(q)


def normalize_label(k):
    k = str(k).strip()
    if len(k) == 1 and k.upper() in list("ABCDE"):
        return k.upper()
    m = re.search(r"\b([A-E])\b", k.upper())
    return m.group(1) if m else k.upper()


def get_options(x):
    opts = pick_first(x, ["options", "choices", "answer_choices", "candidate_answers", "candidates"], None)

    out = {}
    if isinstance(opts, dict):
        for k, v in opts.items():
            lab = normalize_label(k)
            out[lab] = str(v)
    elif isinstance(opts, list):
        for i, v in enumerate(opts):
            lab = chr(ord("A") + i)
            if isinstance(v, dict):
                text = pick_first(v, ["text", "label", "answer", "content"], str(v))
            else:
                text = v
            out[lab] = str(text)

    if not out:
        for lab in list("ABCDE"):
            for key in [lab, lab.lower(), f"option_{lab}", f"option_{lab.lower()}"]:
                if key in x:
                    out[lab] = str(x[key])

    return dict(sorted(out.items()))


def get_gold_label(x, opts):
    gold = pick_first(x, [
        "gold", "answer", "answer_idx", "answer_idx_letter", "gold_answer",
        "correct_answer", "label", "target", "final_answer"
    ], None)

    if gold is None:
        return None

    gs = str(gold).strip()
    lab = normalize_label(gs)
    if lab in opts:
        return lab

    for k, v in opts.items():
        if str(v).strip().lower() == gs.lower():
            return k

    # PubMedQA sometimes has yes/no/maybe labels.
    if gs.lower() in ["yes", "no", "maybe"]:
        for k, v in opts.items():
            if str(v).strip().lower() == gs.lower():
                return k

    return lab if lab in list("ABCDE") else None


def format_options(opts):
    return "\n".join([f"{k}. {v}" for k, v in opts.items()])


def make_hypmed_prompt(question, opts):
    return f"""You are answering a biomedical multiple-choice question.

Evaluate whether the answer space is valid before selecting an answer.

SPACE labels:
- VALID: the provided options contain one medically supported best answer.
- INCOMPLETE: the correct answer appears missing or insufficiently represented by the options.
- CONTRADICTED: the options are internally inconsistent, duplicated in a way that prevents a unique answer, or mutually contradictory.

Return ONLY one JSON object with exactly these keys:
{{"space_label": "VALID|INCOMPLETE|CONTRADICTED", "answer": "A|B|C|D|E", "confidence": 0.0}}

Question:
{question}

Options:
{format_options(opts)}
"""


def build_space_stress(n_per_dataset=60, seed=42):
    rng = random.Random(seed)
    examples = []

    for ds, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

        rows = read_jsonl(path)
        usable = []
        for i, r in enumerate(rows):
            q = get_question(r)
            opts = get_options(r)
            gold = get_gold_label(r, opts)
            if q and opts and gold in opts and len(opts) >= 3:
                usable.append((i, r, q, opts, gold))

        rng.shuffle(usable)
        usable = usable[:n_per_dataset]

        for i, r, q, opts, gold in usable:
            base_id = pick_first(r, ["id", "question_id", "uid"], f"{ds}_{i}")

            # VALID: original item.
            examples.append({
                "id": f"{base_id}_valid",
                "dataset": ds,
                "corruption": "valid_original",
                "gold_space": "VALID",
                "gold_answer_original": gold,
                "question": q,
                "options": opts,
                "prompt": make_hypmed_prompt(q, opts),
            })

            # INCOMPLETE: remove correct option and keep the remaining option set.
            inc_opts = {k: v for k, v in opts.items() if k != gold}
            examples.append({
                "id": f"{base_id}_incomplete",
                "dataset": ds,
                "corruption": "correct_option_removed",
                "gold_space": "INCOMPLETE",
                "gold_answer_original": gold,
                "question": q,
                "options": inc_opts,
                "prompt": make_hypmed_prompt(q, inc_opts),
            })

            # CONTRADICTED/ambiguous: duplicate the correct option text under another label,
            # making the answer space non-unique.
            dup_opts = dict(opts)
            non_gold = [k for k in dup_opts if k != gold]
            replace_lab = non_gold[0]
            dup_opts[replace_lab] = dup_opts[gold]
            examples.append({
                "id": f"{base_id}_contradicted",
                "dataset": ds,
                "corruption": "duplicated_correct_option",
                "gold_space": "CONTRADICTED",
                "gold_answer_original": gold,
                "question": q,
                "options": dup_opts,
                "prompt": make_hypmed_prompt(q, dup_opts),
            })

    out = OUT / "space_stress_test_inputs.jsonl"
    write_jsonl(out, examples)
    print(f"Saved SPACE stress-test inputs: {out}")
    print(f"Total examples: {len(examples)}")
    print(pd.Series([e["gold_space"] for e in examples]).value_counts().to_string())
    return examples


def parse_first_json(text):
    if text is None:
        return {}
    text = str(text).replace("Ġ", " ").replace("Ċ", "\n")
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        s = m.group(0)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def normalize_space(x):
    x = str(x or "").upper()
    if "VALID" in x and "IN" not in x:
        return "VALID"
    if "INCOMPLETE" in x:
        return "INCOMPLETE"
    if "CONTRADICT" in x:
        return "CONTRADICTED"
    return ""


def run_space_models(model_scope="primary", n_per_dataset=60):
    examples = build_space_stress(n_per_dataset=n_per_dataset)

    from vllm import LLM, SamplingParams

    models = PRIMARY_MODELS if model_scope == "primary" else ALL_MODELS

    for model_key, model_id in models:
        print("\n" + "=" * 90)
        print(f"Running SPACE stress test: {model_key} :: {model_id}")
        print("=" * 90)

        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            tensor_parallel_size=1,
        )
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            stop=["\n\nQuestion:", "\nQuestion:"],
        )

        prompts = [e["prompt"] for e in examples]
        outputs = llm.generate(prompts, sampling)

        rows = []
        for e, o in zip(examples, outputs):
            raw = o.outputs[0].text if o.outputs else ""
            obj = parse_first_json(raw)
            pred_space = normalize_space(obj.get("space_label"))
            rows.append({
                **{k: e[k] for k in ["id", "dataset", "corruption", "gold_space", "gold_answer_original"]},
                "model": model_key,
                "pred_space": pred_space,
                "pred_answer": obj.get("answer", ""),
                "confidence": obj.get("confidence", None),
                "raw_output": raw,
            })

        out = OUT / f"{model_key}_space_stress_outputs.jsonl"
        write_jsonl(out, rows)
        print(f"Saved: {out}")

        summarize_space(rows, model_key)


def summarize_space(rows, model_key):
    labels = ["VALID", "INCOMPLETE", "CONTRADICTED"]
    df = pd.DataFrame(rows)
    df["covered"] = df["pred_space"].isin(labels)
    df["correct"] = df["pred_space"] == df["gold_space"]

    summary = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        summary.append({
            "model": model_key,
            "dataset": ds,
            "n": len(sub),
            "space_coverage": round(float(sub["covered"].mean()), 4),
            "space_accuracy": round(float(sub["correct"].mean()), 4),
        })
    summary.append({
        "model": model_key,
        "dataset": "ALL",
        "n": len(df),
        "space_coverage": round(float(df["covered"].mean()), 4),
        "space_accuracy": round(float(df["correct"].mean()), 4),
    })

    per_label = []
    for lab in labels:
        sub = df[df["gold_space"] == lab]
        tp = ((df["gold_space"] == lab) & (df["pred_space"] == lab)).sum()
        fp = ((df["gold_space"] != lab) & (df["pred_space"] == lab)).sum()
        fn = ((df["gold_space"] == lab) & (df["pred_space"] != lab)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append({
            "model": model_key,
            "label": lab,
            "support": len(sub),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    s1 = pd.DataFrame(summary)
    s2 = pd.DataFrame(per_label)

    s1_path = OUT / f"{model_key}_space_stress_summary.csv"
    s2_path = OUT / f"{model_key}_space_stress_per_label.csv"
    s1.to_csv(s1_path, index=False)
    s2.to_csv(s2_path, index=False)

    print("\nSPACE stress summary")
    print(s1.to_string(index=False))
    print("\nSPACE stress per-label")
    print(s2.to_string(index=False))


def extract_answer(row):
    for k in ["pred_answer", "answer", "pred", "prediction", "final_answer"]:
        if k in row and row[k] not in [None, ""]:
            s = str(row[k]).strip().upper()
            m = re.search(r"\b([A-E])\b", s)
            return m.group(1) if m else s[:1]
    raw = row.get("raw_output") or row.get("output") or row.get("text") or ""
    obj = parse_first_json(raw)
    if obj.get("answer"):
        return extract_answer({"pred_answer": obj.get("answer")})
    m = re.search(r"(?:answer|final)\s*[:=]\s*([A-E])", str(raw), flags=re.I)
    return m.group(1).upper() if m else ""


def extract_gold(row):
    for k in ["gold", "gold_answer", "reference", "label", "answer_key", "target"]:
        if k in row and row[k] not in [None, ""]:
            s = str(row[k]).strip().upper()
            m = re.search(r"\b([A-E])\b", s)
            return m.group(1) if m else s[:1]
    return ""


def load_pred_file(path):
    rows = read_jsonl(path)
    out = {}
    for i, r in enumerate(rows):
        rid = str(pick_first(r, ["id", "question_id", "uid"], i))
        pred = extract_answer(r)
        gold = extract_gold(r)
        if not gold:
            # Try nested/common names.
            gold = extract_gold(r.get("example", {})) if isinstance(r.get("example"), dict) else ""
        out[rid] = {"pred": pred, "gold": gold, "raw": r}
    return out


def exact_mcnemar_p(b, c):
    # Exact two-sided binomial over discordant pairs.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def paired_significance():
    print("\n" + "=" * 90)
    print("Paired McNemar tests")
    print("=" * 90)

    pairs = [
        ("qwen2_5_7b_instruct", "cot"),
        ("microsoft_phi_4_mini_instruct", "cot"),
    ]

    rows = []
    for model, baseline in pairs:
        for ds in ["medqa", "medmcqa", "pubmedqa"]:
            prop_path = ROOT / f"results/fusion/{model}_fusion_majority_answer_hypmed_v3_space_{ds}_original1000.jsonl"
            base_path = ROOT / f"results/{model}_{baseline}_{ds}_original1000.jsonl"
            if not prop_path.exists() or not base_path.exists():
                print(f"Missing pair: {prop_path} or {base_path}")
                continue

            prop = load_pred_file(prop_path)
            base = load_pred_file(base_path)

            common = sorted(set(prop) & set(base))
            b = c = both_correct = both_wrong = 0
            for rid in common:
                gold = prop[rid]["gold"] or base[rid]["gold"]
                if not gold:
                    continue
                pc = prop[rid]["pred"] == gold
                bc = base[rid]["pred"] == gold
                if pc and bc:
                    both_correct += 1
                elif (not pc) and (not bc):
                    both_wrong += 1
                elif pc and not bc:
                    b += 1
                elif bc and not pc:
                    c += 1

            rows.append({
                "model": model,
                "dataset": ds,
                "baseline": baseline,
                "n_shared": both_correct + both_wrong + b + c,
                "proposed_correct_baseline_wrong": b,
                "baseline_correct_proposed_wrong": c,
                "mcnemar_exact_p": exact_mcnemar_p(b, c),
            })

    df = pd.DataFrame(rows)
    out = OUT / "mcnemar_tests_qwen_phi.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {out}")


def fallback_sensitivity():
    print("\n" + "=" * 90)
    print("Fallback-order sensitivity")
    print("=" * 90)

    methods = ["cot", "direct", "hypothesismed_v3"]
    method_file = {
        "cot": "{model}_cot_{ds}_original1000.jsonl",
        "direct": "{model}_direct_{ds}_original1000.jsonl",
        "hypothesismed_v3": "{model}_hypothesismed_v3_{ds}_original1000.jsonl",
    }

    models = ["qwen2_5_7b_instruct", "microsoft_phi_4_mini_instruct"]
    rows = []

    for model in models:
        for order in itertools.permutations(methods):
            total = correct = parsed = 0
            for ds in ["medqa", "medmcqa", "pubmedqa"]:
                preds = {}
                for m in methods:
                    path = ROOT / "results" / method_file[m].format(model=model, ds=ds)
                    if not path.exists():
                        preds[m] = {}
                    else:
                        preds[m] = load_pred_file(path)

                common = set.intersection(*[set(preds[m].keys()) for m in methods if preds[m]])
                for rid in common:
                    gold = ""
                    answers = []
                    for m in methods:
                        p = preds[m].get(rid, {})
                        if not gold:
                            gold = p.get("gold", "")
                        if p.get("pred"):
                            answers.append((m, p["pred"]))

                    if not gold:
                        continue

                    total += 1
                    counts = Counter([a for _, a in answers])
                    if counts:
                        top_count = max(counts.values())
                        top_answers = [a for a, n in counts.items() if n == top_count]
                    else:
                        top_answers = []

                    if len(top_answers) == 1 and top_count >= 2:
                        final = top_answers[0]
                    else:
                        final = ""
                        for m in order:
                            for mm, a in answers:
                                if mm == m and a:
                                    final = a
                                    break
                            if final:
                                break

                    if final:
                        parsed += 1
                    if final == gold:
                        correct += 1

            rows.append({
                "model": model,
                "fallback_order": ">".join(order),
                "n": total,
                "parse_coverage": round(parsed / total if total else 0, 4),
                "accuracy": round(correct / total if total else 0, 4),
            })

    df = pd.DataFrame(rows).sort_values(["model", "accuracy"], ascending=[True, False])
    out = OUT / "fallback_order_sensitivity.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {out}")


def calibration_ece():
    print("\n" + "=" * 90)
    print("Confidence calibration / ECE")
    print("=" * 90)

    csv_path = ROOT / "results/final_png_only_assets/tables/confidence_rows_from_fusion_outputs.csv"
    if not csv_path.exists():
        print(f"Missing confidence rows CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("Confidence CSV columns:", list(df.columns))

    conf_col = None
    correct_col = None
    dataset_col = None
    model_col = None

    for c in df.columns:
        lc = c.lower()
        if lc in ["confidence", "conf"]:
            conf_col = c
        if lc in ["correct", "is_correct", "answer_correct"]:
            correct_col = c
        if lc == "dataset":
            dataset_col = c
        if lc == "model":
            model_col = c

    if conf_col is None:
        print("Could not find confidence column.")
        return

    if correct_col is None:
        # Try reconstruct from pred/gold columns.
        pred_col = next((c for c in df.columns if c.lower() in ["pred", "pred_answer", "prediction"]), None)
        gold_col = next((c for c in df.columns if c.lower() in ["gold", "gold_answer", "label"]), None)
        if pred_col and gold_col:
            df["_correct"] = df[pred_col].astype(str).str.upper().str[0] == df[gold_col].astype(str).str.upper().str[0]
            correct_col = "_correct"
        else:
            print("Could not find correct column or pred/gold columns.")
            return

    df["_conf"] = pd.to_numeric(df[conf_col], errors="coerce")
    df["_correct"] = df[correct_col].astype(float)
    df = df.dropna(subset=["_conf", "_correct"])
    df = df[(df["_conf"] >= 0) & (df["_conf"] <= 1)]

    bins = [i / 10 for i in range(11)]
    df["_bin"] = pd.cut(df["_conf"], bins=bins, include_lowest=True, right=True)

    rows = []
    ece = 0.0
    n = len(df)
    for b, sub in df.groupby("_bin", observed=False):
        if len(sub) == 0:
            continue
        avg_conf = float(sub["_conf"].mean())
        acc = float(sub["_correct"].mean())
        weight = len(sub) / n
        ece += weight * abs(acc - avg_conf)
        rows.append({
            "bin": str(b),
            "n": len(sub),
            "mean_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
            "abs_gap": round(abs(acc - avg_conf), 4),
        })

    bin_df = pd.DataFrame(rows)
    bin_path = OUT / "confidence_bin_calibration.csv"
    bin_df.to_csv(bin_path, index=False)

    print("\nOverall confidence-bin calibration")
    print(bin_df.to_string(index=False))
    print(f"ECE_10bin: {ece:.4f}")
    print(f"Saved: {bin_path}")

    group_cols = [c for c in [dataset_col, model_col] if c]
    if group_cols:
        group_rows = []
        for keys, sub in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            local = {"n": len(sub)}
            for c, v in zip(group_cols, keys):
                local[c] = v
            local["mean_confidence"] = round(float(sub["_conf"].mean()), 4)
            local["accuracy"] = round(float(sub["_correct"].mean()), 4)
            group_rows.append(local)

        gdf = pd.DataFrame(group_rows)
        gpath = OUT / "confidence_summary_by_group.csv"
        gdf.to_csv(gpath, index=False)
        print("\nConfidence summary by group")
        print(gdf.to_string(index=False))
        print(f"Saved: {gpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space_stress", action="store_true")
    ap.add_argument("--model_scope", choices=["primary", "all"], default="primary")
    ap.add_argument("--n_per_dataset", type=int, default=60)
    ap.add_argument("--analysis_only", action="store_true")
    args = ap.parse_args()

    if args.space_stress and not args.analysis_only:
        run_space_models(model_scope=args.model_scope, n_per_dataset=args.n_per_dataset)

    calibration_ece()
    paired_significance()
    fallback_sensitivity()

    print("\n" + "=" * 90)
    print("DONE. Review-requested outputs saved in:")
    print(OUT)
    print("=" * 90)
    for p in sorted(OUT.glob("*")):
        print(p)


if __name__ == "__main__":
    main()

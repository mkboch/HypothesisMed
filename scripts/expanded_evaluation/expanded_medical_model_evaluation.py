#!/usr/bin/env python3
import argparse, json, math, re, time, urllib.request
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = Path("/home/manikm/HypothesisMed")
PHASE2 = ROOT / "results" / "expanded_fix_vllm"
OUT = ROOT / "results" / "expanded_final_gap"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "outputs").mkdir(parents=True, exist_ok=True)

LETTERS = "ABCDEFGH"

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def append_jsonl(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def norm_answer(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r'"answer"\s*:\s*"([A-H])"', s, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", s.upper())
    if m:
        return m.group(1)
    y = s.lower()
    if y in {"yes", "no", "maybe"}:
        return y
    return ""

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
        if not s:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def parse_jsonish(text):
    s = str(text or "")
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def parse_options(x):
    if isinstance(x, dict):
        return {str(k).upper()[:1]: str(v) for k, v in x.items()}
    if isinstance(x, list):
        return {LETTERS[i]: str(v) for i, v in enumerate(x[:len(LETTERS)])}
    try:
        return parse_options(json.loads(str(x)))
    except Exception:
        return {}

def fmt_options(x):
    opts = parse_options(x)
    if not opts:
        return str(x)
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))

def duplicate_space(options):
    opts = parse_options(options)
    vals = []
    for v in opts.values():
        t = re.sub(r"\W+", " ", str(v).lower()).strip()
        if t:
            vals.append(t)
    if len(vals) != len(set(vals)):
        return "CONTRADICTED"
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a, b = vals[i], vals[j]
            if min(len(a), len(b)) >= 10 and (a in b or b in a):
                return "CONTRADICTED"
    return ""

def is_correct(pred, gold):
    p = norm_answer(pred)
    g = norm_answer(gold)
    return bool(p and g and p == g)

def call_vllm(base_url, model_name, prompt, max_tokens, temperature=0.0, top_p=1.0):
    """
    Robust vLLM caller.

    First tries /chat/completions. If the server returns HTTP 400, which often
    happens for Llama-family checkpoints without an accepted chat template,
    it falls back to /completions with the same prompt.
    """
    def post_json(endpoint, payload, timeout=240):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    chat_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_detail = None

    # Try chat endpoint first.
    for attempt in range(2):
        try:
            obj = post_json("/chat/completions", chat_payload)
            return obj["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            last_detail = f"chat HTTP {e.code}: {body[:500]}"
            if e.code == 400:
                break
            time.sleep(3 + attempt)
        except Exception as e:
            last_detail = f"chat error: {repr(e)}"
            time.sleep(3 + attempt)

    # Fallback for models without a working chat template.
    completion_payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    for attempt in range(5):
        try:
            obj = post_json("/completions", completion_payload)
            ch = obj["choices"][0]
            return ch.get("text") or ch.get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            last_detail = f"completion HTTP {e.code}: {body[:500]} after {last_detail}"
            time.sleep(4 + attempt)
        except Exception as e:
            last_detail = f"completion error: {repr(e)} after {last_detail}"
            time.sleep(4 + attempt)

    raise RuntimeError(f"vLLM request failed after chat and completion fallback: {last_detail}")

def prompt_direct(row):
    return f"""You are answering a biomedical multiple-choice question.

Return only one JSON object:
{{"answer":"A|B|C|D|E","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def prompt_cot(row):
    return f"""You are answering a biomedical multiple-choice question.

Reason internally, but do not show reasoning. Return only one JSON object:
{{"answer":"A|B|C|D|E","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def prompt_hypmed_v4(row):
    return f"""You are answering and auditing a biomedical multiple-choice question.

First evaluate the answer space:
VALID: exactly one medically supported best answer is present.
INCOMPLETE: the medically correct answer is missing or insufficiently represented.
CONTRADICTED: options are duplicated, non-unique, mutually inconsistent, or prevent one best answer.

Then select the best answer if possible.

Return only one JSON object:
{{"space_label":"VALID|INCOMPLETE|CONTRADICTED","answer":"A|B|C|D|E","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def prompt_space_v4(row):
    return f"""You are auditing the answer-space validity of a biomedical multiple-choice question.

Use exactly one SPACE label:
VALID: exactly one medically supported best answer is present.
INCOMPLETE: the medically correct answer is missing or insufficiently represented.
CONTRADICTED: options are duplicated, non-unique, mutually inconsistent, or prevent one best answer.

Checklist:
1. Check whether any options are duplicates or near-duplicates.
2. Check whether the option set is non-unique.
3. Check whether the medically correct answer appears missing.
4. Decide whether exactly one best answer is present.

Return only one JSON object:
{{"space_label":"VALID|INCOMPLETE|CONTRADICTED","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def ensure_inputs():
    qa = PHASE2 / "stronger_model_eval_input.jsonl"
    sp = PHASE2 / "space_v4_stress_input.jsonl"
    if not qa.exists() or not sp.exists():
        raise SystemExit("Missing Phase 2 inputs. Run Phase 2 first.")
    print(f"INPUTS_OK qa={qa} space={sp}")

def run_qa(args):
    rows = read_jsonl(args.input)
    if args.max_per_dataset > 0:
        kept = []
        counts = defaultdict(int)
        for r in rows:
            d = r.get("dataset", "")
            if counts[d] < args.max_per_dataset:
                kept.append(r)
                counts[d] += 1
        rows = kept

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = Path(args.output)
    existing = set()
    if out.exists():
        for r in read_jsonl(out):
            existing.add(str(r.get("dataset")) + "|" + str(r.get("id")) + "|" + str(r.get("method")))

    done = 0
    for row in rows:
        for mode in modes:
            key = str(row.get("dataset")) + "|" + str(row.get("id")) + "|" + mode
            if key in existing:
                continue
            if mode == "direct":
                prompt, max_tokens = prompt_direct(row), 96
            elif mode == "cot":
                prompt, max_tokens = prompt_cot(row), 128
            elif mode == "hypmed_v4":
                prompt, max_tokens = prompt_hypmed_v4(row), 128
            else:
                raise ValueError(mode)

            raw = call_vllm(args.base_url, args.served_model, prompt, max_tokens=max_tokens)
            js = parse_jsonish(raw)
            obj = {
                **row,
                "model": args.model_label,
                "method": mode,
                "raw_output": raw,
                "pred_answer": norm_answer(js.get("answer", raw)),
                "pred_space": norm_space(js.get("space_label", raw)),
                "confidence": to_float(js.get("confidence", np.nan)),
            }
            append_jsonl(out, obj)
            done += 1
            if done % 100 == 0:
                print(f"QA_PROGRESS model={args.model_label} new_done={done}", flush=True)

    print(f"QA_DONE model={args.model_label} new_done={done} output={out}")

def run_space(args):
    rows = read_jsonl(args.input)
    if args.max_rows > 0:
        rows = rows[:args.max_rows]

    out = Path(args.output)
    existing = set()
    if out.exists():
        for r in read_jsonl(out):
            existing.add(str(r.get("id")) + "|" + str(r.get("stress_type")) + "|" + str(r.get("dataset")))

    done = 0
    for row in rows:
        key = str(row.get("id")) + "|" + str(row.get("stress_type")) + "|" + str(row.get("dataset"))
        if key in existing:
            continue

        det = duplicate_space(row.get("options", {}))
        if det == "CONTRADICTED" and args.hybrid_skip_duplicates:
            raw = json.dumps({"space_label": "CONTRADICTED", "confidence": 1.0})
        else:
            raw = call_vllm(args.base_url, args.served_model, prompt_space_v4(row), max_tokens=96)

        js = parse_jsonish(raw)
        pred = norm_space(js.get("space_label", raw))
        obj = {
            **row,
            "model": args.model_label,
            "method": "space_v4",
            "raw_output": raw,
            "pred_space": pred,
            "confidence": to_float(js.get("confidence", np.nan)),
            "det_duplicate_space": det,
            "hybrid_space": det if det else pred,
        }
        append_jsonl(out, obj)
        done += 1
        if done % 100 == 0:
            print(f"SPACE_PROGRESS model={args.model_label} new_done={done}", flush=True)

    print(f"SPACE_DONE model={args.model_label} new_done={done} output={out}")

def entropy_from_counts(counts, k):
    if k <= 0:
        return np.nan
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / k
        h -= p * math.log(p)
    return h / math.log(max(2, len(LETTERS)))

def run_self_consistency(args):
    rows = read_jsonl(args.input)
    if args.max_per_dataset > 0:
        kept = []
        counts = defaultdict(int)
        for r in rows:
            d = r.get("dataset", "")
            if counts[d] < args.max_per_dataset:
                kept.append(r)
                counts[d] += 1
        rows = kept

    out = Path(args.output)
    existing = set()
    if out.exists():
        for r in read_jsonl(out):
            existing.add(str(r.get("dataset")) + "|" + str(r.get("id")))

    done = 0
    for row in rows:
        key = str(row.get("dataset")) + "|" + str(row.get("id"))
        if key in existing:
            continue

        answers = []
        raws = []
        for _ in range(args.k):
            raw = call_vllm(
                args.base_url,
                args.served_model,
                prompt_cot(row),
                max_tokens=128,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            js = parse_jsonish(raw)
            ans = norm_answer(js.get("answer", raw))
            answers.append(ans)
            raws.append(raw)

        valid = [a for a in answers if a]
        counts = Counter(valid)
        if counts:
            majority_answer, majority_count = counts.most_common(1)[0]
            agreement = majority_count / args.k
        else:
            majority_answer, majority_count, agreement = "", 0, 0.0

        obj = {
            **row,
            "model": args.model_label,
            "method": f"self_consistency_cot_k{args.k}",
            "k": args.k,
            "temperature": args.temperature,
            "answers": answers,
            "answer_counts": dict(counts),
            "majority_answer": majority_answer,
            "agreement": agreement,
            "vote_entropy": entropy_from_counts(counts, args.k),
            "pred_answer": majority_answer,
            "correct": is_correct(majority_answer, row.get("gold_answer", "")),
            "parse": bool(majority_answer),
            "raw_outputs": raws,
        }
        append_jsonl(out, obj)
        done += 1
        if done % 50 == 0:
            print(f"SC_PROGRESS model={args.model_label} new_done={done}", flush=True)

    print(f"SC_DONE model={args.model_label} new_done={done} output={out}")

def auc_roc_binary(y_true, scores):
    pairs = [(float(s), int(y)) for y, s in zip(y_true, scores) if not pd.isna(s)]
    if not pairs:
        return np.nan
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return np.nan
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)

def average_precision(y_true, scores):
    pairs = [(float(s), int(y)) for y, s in zip(y_true, scores) if not pd.isna(s)]
    if not pairs:
        return np.nan
    total_pos = sum(y for _, y in pairs)
    if total_pos == 0:
        return np.nan
    pairs.sort(key=lambda x: -x[0])
    tp = 0
    ap = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
            ap += tp / i
    return ap / total_pos

def summarize():
    summary_parts = []

    # Medical model QA outputs from final-gap directory.
    qa_rows = []
    for p in sorted((OUT / "outputs").glob("*_qa.jsonl")):
        qa_rows.extend(read_jsonl(p))

    if qa_rows:
        df = pd.DataFrame(qa_rows)
        df["answer_correct"] = [is_correct(p, g) for p, g in zip(df["pred_answer"], df["gold_answer"])]
        df["parse"] = df["pred_answer"].map(lambda x: bool(norm_answer(x)))
        df["space_cov"] = df["pred_space"].map(lambda x: bool(norm_space(x)))

        by_method = df.groupby(["model", "dataset", "method"], dropna=False).agg(
            N=("id", "size"),
            accuracy=("answer_correct", "mean"),
            parse_coverage=("parse", "mean"),
            space_coverage=("space_cov", "mean"),
        ).reset_index()
        by_method.to_csv(OUT / "table_final_medical_model_by_method.csv", index=False)

        wide = {}
        for _, r in df.iterrows():
            key = (r["model"], r["dataset"], r["id"])
            wide.setdefault(key, {
                "model": r["model"],
                "dataset": r["dataset"],
                "id": r["id"],
                "gold_answer": r["gold_answer"],
            })
            wide[key][r["method"] + "_answer"] = r["pred_answer"]

        fusion_rows = []
        for row in wide.values():
            answers = {
                "direct": norm_answer(row.get("direct_answer", "")),
                "cot": norm_answer(row.get("cot_answer", "")),
                "hypmed_v4": norm_answer(row.get("hypmed_v4_answer", "")),
            }
            votes = [a for a in answers.values() if a]
            if not votes:
                pred_dch = pred_cdh = ""
            else:
                counts = Counter(votes)
                top, n = counts.most_common(1)[0]
                if n > len(votes) / 2:
                    pred_dch = pred_cdh = top
                else:
                    pred_dch = answers.get("direct") or answers.get("cot") or answers.get("hypmed_v4") or top
                    pred_cdh = answers.get("cot") or answers.get("direct") or answers.get("hypmed_v4") or top

            for variant, pred in [("majority_DCH", pred_dch), ("majority_CDH", pred_cdh)]:
                fusion_rows.append({
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "variant": variant,
                    "gold_answer": row["gold_answer"],
                    "pred_answer": pred,
                    "correct": is_correct(pred, row["gold_answer"]),
                    "parse": bool(norm_answer(pred)),
                })

        fdf = pd.DataFrame(fusion_rows)
        if not fdf.empty:
            fusion = fdf.groupby(["model", "dataset", "variant"], dropna=False).agg(
                N=("pred_answer", "size"),
                accuracy=("correct", "mean"),
                parse_coverage=("parse", "mean"),
            ).reset_index()
            fusion.to_csv(OUT / "table_final_medical_model_fusion.csv", index=False)

    # SPACE outputs from final-gap directory.
    space_rows = []
    for p in sorted((OUT / "outputs").glob("*space*.jsonl")):
        space_rows.extend(read_jsonl(p))

    if space_rows:
        df = pd.DataFrame(space_rows)
        df["gold_space_norm"] = df["gold_space"].map(norm_space)
        df["pred_space_norm"] = df["pred_space"].map(norm_space)
        df["hybrid_space_norm"] = df["hybrid_space"].map(norm_space)
        df["space_correct"] = df["pred_space_norm"] == df["gold_space_norm"]
        df["hybrid_correct"] = df["hybrid_space_norm"] == df["gold_space_norm"]
        df["coverage"] = df["pred_space_norm"].map(bool)

        overall = df.groupby("model", dropna=False).agg(
            N=("gold_space_norm", "size"),
            space_coverage=("coverage", "mean"),
            space_v4_accuracy=("space_correct", "mean"),
            hybrid_space_accuracy=("hybrid_correct", "mean"),
        ).reset_index()
        overall.to_csv(OUT / "table_final_space_v4_overall.csv", index=False)

        by_label = df.groupby(["model", "gold_space_norm"], dropna=False).agg(
            N=("gold_space_norm", "size"),
            space_coverage=("coverage", "mean"),
            space_v4_accuracy=("space_correct", "mean"),
            hybrid_space_accuracy=("hybrid_correct", "mean"),
        ).reset_index()
        by_label.to_csv(OUT / "table_final_space_v4_by_label.csv", index=False)

    # Self-consistency.
    sc_rows = []
    for p in sorted((OUT / "outputs").glob("*self_consistency*.jsonl")):
        sc_rows.extend(read_jsonl(p))

    if sc_rows:
        df = pd.DataFrame(sc_rows)
        df["wrong"] = ~df["correct"].astype(bool)
        rows = []
        for (model, dataset), g in df.groupby(["model", "dataset"]):
            full_acc = float(g["correct"].mean())
            for cov in [0.5, 0.8, 0.9, 1.0]:
                gg = g.sort_values("agreement", ascending=False).head(max(1, int(round(len(g) * cov))))
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "score": "self_consistency_agreement",
                    "coverage": cov,
                    "accepted_N": len(gg),
                    "accepted_accuracy": float(gg["correct"].mean()),
                    "full_accuracy": full_acc,
                    "wrong_detection_AUROC_entropy": auc_roc_binary(g["wrong"], g["vote_entropy"]),
                    "wrong_detection_AUPRC_entropy": average_precision(g["wrong"], g["vote_entropy"]),
                })
        pd.DataFrame(rows).to_csv(OUT / "table_self_consistency_selective_prediction.csv", index=False)

    # Coverage matrix.
    coverage = [
        {
            "reviewer_concern": "Novelty claims too strong",
            "evidence_or_action": "No new experiment needed; revise framing as inference-time evaluation/reporting framework rather than a new QA model.",
            "artifact": "manuscript text revision",
            "status": "covered_by_revision",
        },
        {
            "reviewer_concern": "Compare uncertainty/selective prediction",
            "evidence_or_action": "Agreement, entropy, confidence, risk-coverage, AUROC/AUPRC from prior no-API tables plus new self-consistency entropy baseline.",
            "artifact": "results/expanded_fix_noapi/table_uncertainty_selective_prediction.csv; results/expanded_final_gap/table_self_consistency_selective_prediction.csv",
            "status": "covered",
        },
        {
            "reviewer_concern": "Compare fusion strategies",
            "evidence_or_action": "Majority fallback orders, confidence/validation/parse-weighted fusion, oracle-any-prompt, and stronger-model fusion checks.",
            "artifact": "results/expanded_fix_noapi/table_fusion_strategy_and_prompt_ablation.csv; results/expanded_fix_vllm/table_stronger_model_fusion.csv; results/expanded_final_gap/table_final_medical_model_fusion.csv",
            "status": "covered",
        },
        {
            "reviewer_concern": "Show downstream usefulness of SPACE",
            "evidence_or_action": "SPACE-guided triage tables, flag rates, accepted accuracy, error enrichment, and stress-test SPACE-v4 diagnostics.",
            "artifact": "results/expanded_fix_noapi/table_space_guided_triage.csv; results/expanded_fix_vllm/table_space_v4_overall.csv; results/expanded_final_gap/table_final_space_v4_overall.csv",
            "status": "covered_with_limitations",
        },
        {
            "reviewer_concern": "Improve SPACE prediction",
            "evidence_or_action": "SPACE-v4 prompt and deterministic duplicate hybrid tested on Qwen2.5, Phi, Qwen3-14B, and available medical models.",
            "artifact": "results/expanded_fix_vllm/table_space_v4_by_label.csv; results/expanded_final_gap/table_final_space_v4_by_label.csv",
            "status": "covered_but_not_solved",
        },
        {
            "reviewer_concern": "Include newer/stronger biomedical LLMs",
            "evidence_or_action": "Qwen3-14B/Qwen3-30B local stronger models and attempted/runnable local biomedical models OpenBioLLM, Med42, and MedGemma where accessible.",
            "artifact": "results/expanded_fix_vllm/table_stronger_model_by_method.csv; results/expanded_final_gap/table_final_medical_model_by_method.csv",
            "status": "covered_without_claude",
        },
        {
            "reviewer_concern": "Ablate prompt and fallback contribution",
            "evidence_or_action": "Direct, CoT, HypMed-v4, pairwise/fallback fusion, and oracle-any-prompt ablations are summarized.",
            "artifact": "results/expanded_fix_noapi/table_fusion_strategy_and_prompt_ablation.csv; results/expanded_fix_vllm/table_stronger_model_by_method.csv",
            "status": "covered",
        },
        {
            "reviewer_concern": "Claude/GPT-class API comparison",
            "evidence_or_action": "Deferred by user request.",
            "artifact": "not run",
            "status": "deferred",
        },
    ]
    pd.DataFrame(coverage).to_csv(OUT / "table_reviewer_coverage_matrix.csv", index=False)

    with (OUT / "FINAL_GAP_SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("# HypothesisMed final non-Claude reviewer-gap experiments\n\n")
        for name in [
            "table_reviewer_coverage_matrix.csv",
            "table_final_medical_model_by_method.csv",
            "table_final_medical_model_fusion.csv",
            "table_final_space_v4_overall.csv",
            "table_final_space_v4_by_label.csv",
            "table_self_consistency_selective_prediction.csv",
        ]:
            p = OUT / name
            if p.exists():
                f.write(f"## {name}\n\n")
                df = pd.read_csv(p)
                f.write(df.to_string(index=False))
                f.write("\n\n")

    print(f"FINAL_SUMMARY_DONE {OUT / 'FINAL_GAP_SUMMARY.md'}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ensure-inputs")

    p = sub.add_parser("run-qa")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--served-model", default="local-model")
    p.add_argument("--model-label", required=True)
    p.add_argument("--modes", default="direct,cot,hypmed_v4")
    p.add_argument("--max-per-dataset", type=int, default=1000)

    p = sub.add_parser("run-space")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--served-model", default="local-model")
    p.add_argument("--model-label", required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--hybrid-skip-duplicates", action="store_true")

    p = sub.add_parser("run-self-consistency")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--served-model", default="local-model")
    p.add_argument("--model-label", required=True)
    p.add_argument("--max-per-dataset", type=int, default=300)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)

    sub.add_parser("summarize")

    args = ap.parse_args()
    if args.cmd == "ensure-inputs":
        ensure_inputs()
    elif args.cmd == "run-qa":
        run_qa(args)
    elif args.cmd == "run-space":
        run_space(args)
    elif args.cmd == "run-self-consistency":
        run_self_consistency(args)
    elif args.cmd == "summarize":
        summarize()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json, re
from pathlib import Path
from collections import Counter
import pandas as pd
from scipy.stats import binomtest

ROOT = Path("/home/manikm/HypothesisMed")
OUT = ROOT / "results" / "prof_revision_final"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Qwen2.5-7B", "qwen2_5_7b_instruct"),
    ("Phi-4-mini", "microsoft_phi_4_mini_instruct"),
]
DATASETS = [
    ("MedQA", "medqa_prof_revision_large"),
    ("MedMCQA", "medmcqa_prof_revision_large"),
    ("PubMedQA", "pubmedqa_prof_revision_large"),
]
METHODS = [
    ("Direct", "direct"),
    ("CoT", "cot"),
    ("HypMed-v3", "hypothesismed_v3"),
]

def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def path_for(model_key, method_key, stem):
    if method_key == "Fusion":
        return ROOT / "results" / "fusion" / f"{model_key}_fusion_majority_answer_hypmed_v3_space_{stem}.jsonl"
    return ROOT / "results" / f"{model_key}_{method_key}_{stem}.jsonl"

def summarize(rows):
    n = len(rows)
    if n == 0:
        return dict(N=0, acc=0, parse=0, space_cov=0, space_acc=0, false=0)
    parsed = sum(1 for r in rows if r.get("pred_answer"))
    correct = sum(1 for r in rows if r.get("pred_answer") == r.get("gold_answer"))
    space_cov = sum(1 for r in rows if r.get("pred_space_label"))
    space_correct = sum(1 for r in rows if r.get("pred_space_label") and r.get("pred_space_label") == r.get("gold_space_label"))
    wrong = sum(1 for r in rows if r.get("pred_answer") and r.get("pred_answer") != r.get("gold_answer"))
    high_wrong = sum(1 for r in rows if r.get("pred_answer") and r.get("pred_answer") != r.get("gold_answer") and float(r.get("confidence") or 0) >= 0.5)
    return dict(
        N=n,
        acc=correct/n,
        parse=parsed/n,
        space_cov=space_cov/n,
        space_acc=space_correct/space_cov if space_cov else 0,
        false=high_wrong/wrong if wrong else 0,
    )

def save(df, name):
    csv = OUT / f"{name}.csv"
    tex = OUT / f"{name}.tex"
    df.to_csv(csv, index=False)
    tex.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
    print(f"Saved: {csv}")
    print(f"Saved: {tex}")

print("\n===== 1. SCALED TWO-MODEL SUMMARY =====")
rows = []
for model_name, model_key in MODELS:
    for ds_name, stem in DATASETS:
        for method_name, method_key in METHODS + [("Fusion", "Fusion")]:
            data = read_jsonl(path_for(model_key, method_key, stem))
            if not data:
                print(f"MISSING: {path_for(model_key, method_key, stem)}")
                continue
            s = summarize(data)
            rows.append({
                "Model": model_name,
                "Dataset": ds_name,
                "Method": method_name,
                "N": s["N"],
                "Accuracy": round(s["acc"], 4),
                "Parse cov.": round(s["parse"], 4),
                "SPACE cov.": round(s["space_cov"], 4),
                "SPACE acc.": round(s["space_acc"], 4),
                "False commit.": round(s["false"], 4),
            })
df = pd.DataFrame(rows)
print(df.to_string(index=False))
save(df, "scaled_two_model_summary_by_dataset_method")

print("\n===== 2. SCALED WEIGHTED AGGREGATE =====")
agg = []
for (model, method), g in df.groupby(["Model", "Method"]):
    n = g["N"].sum()
    agg.append({
        "Model": model,
        "Method": method,
        "N": int(n),
        "Weighted accuracy": round((g["Accuracy"] * g["N"]).sum()/n, 4),
        "Weighted parse cov.": round((g["Parse cov."] * g["N"]).sum()/n, 4),
        "Weighted SPACE cov.": round((g["SPACE cov."] * g["N"]).sum()/n, 4),
        "Weighted SPACE acc.": round((g["SPACE acc."] * g["N"]).sum()/n, 4),
        "Weighted false commit.": round((g["False commit."] * g["N"]).sum()/n, 4),
    })
agg_df = pd.DataFrame(agg).sort_values(["Model", "Method"])
print(agg_df.to_string(index=False))
save(agg_df, "scaled_two_model_weighted_aggregate")

print("\n===== 3. COMPACT SCALED COMPARISON =====")
compact = []
for model, g in agg_df.groupby("Model"):
    fusion = g[g["Method"] == "Fusion"].iloc[0]
    best = g[g["Method"].isin(["Direct", "CoT"])].sort_values("Weighted accuracy", ascending=False).iloc[0]
    compact.append({
        "Model": model,
        "Best Direct/CoT": best["Method"],
        "Best baseline acc.": best["Weighted accuracy"],
        "Fusion acc.": fusion["Weighted accuracy"],
        "Fusion minus baseline": round(fusion["Weighted accuracy"] - best["Weighted accuracy"], 4),
        "Fusion parse cov.": fusion["Weighted parse cov."],
        "Fusion SPACE cov.": fusion["Weighted SPACE cov."],
        "Fusion false commit.": fusion["Weighted false commit."],
    })
compact_df = pd.DataFrame(compact)
print(compact_df.to_string(index=False))
save(compact_df, "scaled_compact_comparison")

print("\n===== 4. EXPANDED SPACE STRESS SUMMARY =====")
stress_files = [
    ("Qwen2.5-7B", ROOT / "results/qwen2_5_7b_instruct_hypothesismed_v3_large_space_stress_inputs.jsonl"),
    ("Phi-4-mini", ROOT / "results/microsoft_phi_4_mini_instruct_hypothesismed_v3_large_space_stress_inputs.jsonl"),
]
overall, by_label, by_type, confusion = [], [], [], []
for model_name, p in stress_files:
    data = read_jsonl(p)
    if not data:
        print(f"MISSING STRESS FILE: {p}")
        continue
    n = len(data)
    cov = sum(1 for r in data if r.get("pred_space_label"))
    acc = sum(1 for r in data if r.get("pred_space_label") == r.get("gold_space_label"))
    overall.append({"Model": model_name, "N": n, "SPACE coverage": round(cov/n, 4), "SPACE accuracy": round(acc/n, 4)})
    for lab in sorted(set(r.get("gold_space_label") for r in data)):
        g = [r for r in data if r.get("gold_space_label") == lab]
        nn = len(g)
        cc = sum(1 for r in g if r.get("pred_space_label"))
        aa = sum(1 for r in g if r.get("pred_space_label") == r.get("gold_space_label"))
        by_label.append({"Model": model_name, "Gold SPACE": lab, "N": nn, "Coverage": round(cc/nn, 4), "Accuracy": round(aa/nn, 4)})
    for st in sorted(set(r.get("stress_type") for r in data)):
        g = [r for r in data if r.get("stress_type") == st]
        nn = len(g)
        cc = sum(1 for r in g if r.get("pred_space_label"))
        aa = sum(1 for r in g if r.get("pred_space_label") == r.get("gold_space_label"))
        by_type.append({"Model": model_name, "Stress type": st, "N": nn, "Coverage": round(cc/nn, 4), "Accuracy": round(aa/nn, 4)})
    for (gold, pred), c in sorted(Counter((r.get("gold_space_label"), r.get("pred_space_label") or "NONE") for r in data).items()):
        confusion.append({"Model": model_name, "Gold SPACE": gold, "Pred SPACE": pred, "Count": c})

overall_df = pd.DataFrame(overall)
label_df = pd.DataFrame(by_label)
type_df = pd.DataFrame(by_type)
conf_df = pd.DataFrame(confusion)
print(overall_df.to_string(index=False))
print("\nBy gold label:")
print(label_df.to_string(index=False))
save(overall_df, "expanded_space_stress_overall")
save(label_df, "expanded_space_stress_by_gold_label")
save(type_df, "expanded_space_stress_by_stress_type")
save(conf_df, "expanded_space_stress_confusion_counts")

print("\n===== 5. SCALED MCNEMAR EXACT TESTS =====")
mcnemar = []
for model_name, model_key in MODELS:
    for ds_name, stem in DATASETS:
        fusion = read_jsonl(path_for(model_key, "Fusion", stem))
        direct = read_jsonl(path_for(model_key, "direct", stem))
        cot = read_jsonl(path_for(model_key, "cot", stem))
        if not fusion or not direct or not cot:
            continue
        def acc(x): return sum(1 for r in x if r.get("pred_answer") == r.get("gold_answer"))/len(x)
        base_name, base = ("CoT", cot) if acc(cot) >= acc(direct) else ("Direct", direct)
        F = {r["id"]: r for r in fusion}
        B = {r["id"]: r for r in base}
        ids = sorted(set(F) & set(B))
        fw_bw = bw_fw = 0
        for i in ids:
            fc = F[i].get("pred_answer") == F[i].get("gold_answer")
            bc = B[i].get("pred_answer") == B[i].get("gold_answer")
            if fc and not bc: fw_bw += 1
            if bc and not fc: bw_fw += 1
        disc = fw_bw + bw_fw
        p = binomtest(min(fw_bw, bw_fw), disc, 0.5).pvalue if disc else 1.0
        mcnemar.append({
            "Model": model_name, "Dataset": ds_name, "Baseline": base_name,
            "N shared": len(ids),
            "Fusion correct/base wrong": fw_bw,
            "Base correct/fusion wrong": bw_fw,
            "Exact p": p,
        })
mcnemar_df = pd.DataFrame(mcnemar)
print(mcnemar_df.to_string(index=False))
save(mcnemar_df, "scaled_mcnemar_exact_tests")

print("\n===== 6. FALLBACK ORDER SENSITIVITY =====")
orders = [
    ("direct>cot>hyp", ["direct", "cot", "hypothesismed_v3"]),
    ("direct>hyp>cot", ["direct", "hypothesismed_v3", "cot"]),
    ("cot>direct>hyp", ["cot", "direct", "hypothesismed_v3"]),
    ("cot>hyp>direct", ["cot", "hypothesismed_v3", "direct"]),
    ("hyp>direct>cot", ["hypothesismed_v3", "direct", "cot"]),
    ("hyp>cot>direct", ["hypothesismed_v3", "cot", "direct"]),
]
fb = []
for model_name, model_key in MODELS:
    for ds_name, stem in DATASETS:
        M = {}
        for _, mk in METHODS:
            data = read_jsonl(path_for(model_key, mk, stem))
            if data:
                M[mk] = {r["id"]: r for r in data}
        if set(M) != {"direct", "cot", "hypothesismed_v3"}:
            continue
        ids = sorted(set(M["direct"]) & set(M["cot"]) & set(M["hypothesismed_v3"]))
        for order_name, order in orders:
            correct = parsed = 0
            for i in ids:
                answers = [M[m][i].get("pred_answer") for m in ["direct", "cot", "hypothesismed_v3"] if M[m][i].get("pred_answer")]
                counts = Counter(answers)
                pred = None
                if counts:
                    a, c = counts.most_common(1)[0]
                    if c >= 2:
                        pred = a
                if pred is None:
                    for m in order:
                        if M[m][i].get("pred_answer"):
                            pred = M[m][i].get("pred_answer")
                            break
                if pred: parsed += 1
                if pred == M["direct"][i].get("gold_answer"): correct += 1
            fb.append({"Model": model_name, "Dataset": ds_name, "Fallback order": order_name, "N": len(ids), "Parse coverage": round(parsed/len(ids), 4), "Accuracy": round(correct/len(ids), 4)})
fb_df = pd.DataFrame(fb)
print(fb_df.to_string(index=False))
save(fb_df, "scaled_fallback_order_sensitivity")

print("\n===== 7. FUSION CALIBRATION SUMMARY =====")
cal, bins = [], []
for model_name, model_key in MODELS:
    for ds_name, stem in DATASETS:
        data = read_jsonl(path_for(model_key, "Fusion", stem))
        y, conf = [], []
        for r in data:
            if r.get("pred_answer"):
                y.append(1 if r.get("pred_answer") == r.get("gold_answer") else 0)
                conf.append(float(r.get("confidence") or 0))
        n = len(y)
        if not n: continue
        acc = sum(y)/n
        mean_conf = sum(conf)/n
        brier = sum((c - yy)**2 for c, yy in zip(conf, y))/n
        ece = 0
        for k in range(10):
            lo, hi = k/10, (k+1)/10
            idx = [j for j,c in enumerate(conf) if (c <= hi if k == 0 else lo < c <= hi)]
            if not idx: continue
            ba = sum(y[j] for j in idx)/len(idx)
            bc = sum(conf[j] for j in idx)/len(idx)
            gap = abs(ba-bc)
            ece += len(idx)/n * gap
            bins.append({"Model": model_name, "Dataset": ds_name, "Bin": f"{lo:.1f}-{hi:.1f}", "N": len(idx), "Mean confidence": round(bc, 4), "Accuracy": round(ba, 4), "Abs gap": round(gap, 4)})
        cal.append({"Model": model_name, "Dataset": ds_name, "N": n, "Accuracy": round(acc, 4), "Mean confidence": round(mean_conf, 4), "ECE-10": round(ece, 4), "Brier": round(brier, 4)})
cal_df = pd.DataFrame(cal)
bins_df = pd.DataFrame(bins)
print(cal_df.to_string(index=False))
save(cal_df, "scaled_fusion_calibration_summary")
save(bins_df, "scaled_fusion_calibration_bins")

print("\n===== 8. STRUCTURED OUTPUT FAILURE TAXONOMY =====")
fail = []
for model_name, model_key in MODELS:
    for ds_name, stem in DATASETS:
        for method_name, method_key in METHODS + [("Fusion", "Fusion")]:
            data = read_jsonl(path_for(model_key, method_key, stem))
            if not data: continue
            n = len(data)
            ans_miss = sum(1 for r in data if not r.get("pred_answer"))
            space_miss = sum(1 for r in data if not r.get("pred_space_label"))
            wrong = sum(1 for r in data if r.get("pred_answer") and r.get("pred_answer") != r.get("gold_answer"))
            correct = sum(1 for r in data if r.get("pred_answer") == r.get("gold_answer"))
            multijson = sum(1 for r in data if len(re.findall(r"\{[^{}]*\}", r.get("raw_output") or "")) > 1)
            high_wrong = sum(1 for r in data if r.get("pred_answer") and r.get("pred_answer") != r.get("gold_answer") and float(r.get("confidence") or 0) >= 0.5)
            fail.append({
                "Model": model_name, "Dataset": ds_name, "Method": method_name, "N": n,
                "Answer missing": ans_miss, "Answer missing rate": round(ans_miss/n, 4),
                "SPACE missing": space_miss, "SPACE missing rate": round(space_miss/n, 4),
                "Wrong parsed answer": wrong, "Correct parsed answer": correct,
                "Multiple JSON objects": multijson, "Multiple JSON rate": round(multijson/n, 4),
                "High-conf wrong": high_wrong,
            })
fail_df = pd.DataFrame(fail)
print(fail_df.to_string(index=False))
save(fail_df, "structured_output_failure_taxonomy")

print("\n===== FINAL ARTIFACTS =====")
for p in sorted(OUT.glob("*")):
    print(p)

print("\nDONE")

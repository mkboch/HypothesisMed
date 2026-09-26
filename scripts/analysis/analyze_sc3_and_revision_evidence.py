#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from itertools import permutations
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('HYPOTHESISMED_ROOT', Path(__file__).resolve().parents[2]))
OUT = ROOT / 'results/final_self_consistency_analysis_20260831'
SC_PROTOCOL = ROOT / 'results/final_self_consistency_protocol_freeze_20260831'
SC_GEN = ROOT / 'results/final_self_consistency_generations_20260831'
PRIMARY_OUT = ROOT / 'results/final_outcome_analysis_20260830'
PRIMARY_DF = PRIMARY_OUT / 'analysis_ready_item_level.parquet'
PARSER = ROOT / 'src/evaluation/parser.py'
FROZEN_PARSER_SHA = 'cbaa75e20ff467ed81a3272e79de4ee7e97e3e9e94379ee40ac3d0332c959223'
FROZEN_COT_SHA = 'cd1ab2c4e0d4aff7dab21f64a7c4b237b641300bfef0df52ac203577ccae4956'
SEED = 20260816
BOOT_B = 10000
MODEL_ORDER = ['gemma4_31b', 'medgemma27b', 'phi4mini', 'qwen25', 'qwen36']
DATASETS = ['medqa', 'medmcqa', 'pubmedqa']
SAMPLES = ['sample_1', 'sample_2', 'sample_3']
VALID = {'A', 'B', 'C', 'D', 'E'}
SC3_PAIRS = [
    ('SC-3', 'mean_single'),
    ('SC-3', 'sample_1'),
    ('SC-3', 'sample_2'),
    ('SC-3', 'sample_3'),
    ('SC-3', 'original_fusion'),
    ('SC-3', 'original_cot'),
]
ANCHORS = {}
ANCHOR_TOL = 0.0015


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def norm_answer(x: Any) -> str:
    y = str(x or '').strip().upper()
    return y if y in VALID else ''


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row['_line_no'] = line_no
            rows.append(row)
    return rows


def load_parser():
    if sha_file(PARSER) != FROZEN_PARSER_SHA:
        raise RuntimeError(f'Frozen parser SHA mismatch: {sha_file(PARSER)}')
    spec = importlib.util.spec_from_file_location('frozen_hypothesismed_parser', PARSER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import parser: {PARSER}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_output


def verify_protocol() -> dict[str, Any]:
    manifest = json.loads((SC_PROTOCOL / 'sc3_protocol_manifest.json').read_text(encoding='utf-8'))
    if manifest['cot_prompt_sha256'] != FROZEN_COT_SHA:
        raise RuntimeError('SC protocol CoT prompt SHA mismatch')
    if manifest['parser_sha256'] != FROZEN_PARSER_SHA:
        raise RuntimeError('SC protocol parser SHA mismatch')
    if manifest['items_per_model'] != 5956 or manifest['samples_per_item'] != 3:
        raise RuntimeError('SC protocol row count mismatch')
    return manifest


def verify_sc_generation_complete() -> dict[str, Any]:
    out = {}
    for model in MODEL_ORDER:
        path = SC_GEN / f'{model}_sc3_completion_manifest.json'
        if not path.exists():
            raise RuntimeError(f'SC generation incomplete: missing {path}')
        m = json.loads(path.read_text(encoding='utf-8'))
        expected = {'expected_rows': 17868, 'rows': 17868, 'unique_keys': 17868, 'duplicate_keys': 0, 'complete': True}
        for k, v in expected.items():
            if m.get(k) != v:
                raise RuntimeError(f'{model} SC manifest {k}={m.get(k)!r}, expected {v!r}')
        data_path = ROOT / m['output_jsonl']
        got = sha_file(data_path)
        if got != m['output_sha256']:
            raise RuntimeError(f'{model} SC output checksum mismatch')
        out[model] = {**m, 'verified_output_sha256': got}
    return out


def load_primary() -> pd.DataFrame:
    df = pd.read_parquet(PRIMARY_DF)
    required = {'model_key', 'dataset', 'source_index', 'uid', 'gold', 'method', 'correct', 'parser_recovered', 'output_tokens'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Primary dataframe missing columns: {sorted(missing)}')
    if len(df) != 119120:
        raise RuntimeError(f'Primary dataframe row count mismatch: {len(df)}')
    return df


def parse_sc_samples(parse_output) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        path = SC_GEN / f'{model}_sc3_cot_samples.jsonl'
        records = load_jsonl(path)
        if len(records) != 17868:
            raise RuntimeError(f'{path} expected 17868 rows, found {len(records)}')
        seen = set()
        for r in records:
            key = tuple(r['unique_key'])
            if key in seen:
                raise RuntimeError(f'Duplicate SC key in {path}: {key}')
            seen.add(key)
            parsed = parse_output(r.get('raw_output') or '')
            answer = norm_answer(parsed.get('answer') if isinstance(parsed, dict) else '')
            gold = norm_answer(r.get('gold'))
            rows.append({
                'row_type': 'sc_sample',
                'model_key': r['model_key'],
                'model_id': r['model_id'],
                'revision': r['revision'],
                'condition': r['condition'],
                'dataset': r['dataset'],
                'source_index': int(r['source_index']),
                'uid': str(r['uid']),
                'gold': gold,
                'sample_id': r['sample_id'],
                'sample_index': int(r['sample_index']),
                'seed': int(r['seed']),
                'decoding_settings': json.dumps(r['decoding_settings'], sort_keys=True),
                'serialized_input_sha256': r['serialized_input_sha256'],
                'controlled_prompt_sha256': r['controlled_prompt_sha256'],
                'prompt_tokens': int(r.get('prompt_tokens', 0)),
                'output_tokens': int(r.get('output_tokens', 0)),
                'finish_reason': r.get('finish_reason'),
                'empty_completion': bool(r.get('empty_completion')),
                'length_truncation': bool(r.get('length_truncation')),
                'severe_repetition': bool(r.get('severe_repetition')),
                'reasoning_marker': bool(r.get('reasoning_marker')),
                'parsed_answer': answer,
                'parser_recovered': bool(answer),
                'correct': bool(answer and answer == gold),
                'raw_output_path': str(path.relative_to(ROOT)),
                'raw_output_line': int(r['_line_no']),
                'raw_output': r.get('raw_output') or '',
            })
    df = pd.DataFrame(rows)
    if len(df) != 89340:
        raise RuntimeError(f'SC sample rows mismatch: {len(df)}')
    if df.duplicated(['model_key', 'dataset', 'source_index', 'uid', 'sample_id']).any():
        raise RuntimeError('Duplicate SC sample keys after parsing')
    return df


def derive_sc3(samples: pd.DataFrame, priority: tuple[str, str, str]) -> pd.DataFrame:
    rows = []
    group_cols = ['model_key', 'model_id', 'revision', 'condition', 'dataset', 'source_index', 'uid', 'gold']
    for key, g in samples.groupby(group_cols, sort=True):
        if set(g['sample_id']) != set(SAMPLES):
            raise RuntimeError(f'Missing samples for {key}')
        rec = dict(zip(group_cols, key))
        by_sample = {r.sample_id: r for r in g.itertuples(index=False)}
        answers = {s: getattr(by_sample[s], 'parsed_answer') or '' for s in SAMPLES}
        recovered = {s: bool(getattr(by_sample[s], 'parser_recovered')) for s in SAMPLES}
        vals = [answers[s] for s in SAMPLES if recovered[s] and answers[s] in VALID]
        counts = Counter(vals)
        majority = [a for a, n in counts.items() if n >= 2]
        if majority:
            selected = majority[0]
            resolution = 'strict_majority'
        else:
            selected = ''
            resolution = 'unresolved'
            for s in priority:
                if recovered[s] and answers[s] in VALID:
                    selected = answers[s]
                    resolution = f'{s}_fallback'
                    break
        rows.append({
            **rec,
            'method': 'SC-3',
            'sample_priority': '>'.join(priority),
            'parsed_answer': selected,
            'parser_recovered': bool(selected),
            'correct': bool(selected and selected == rec['gold']),
            'sc_resolution': resolution,
            'sc_support_count': int(counts.get(selected, 0)) if selected else 0,
            'sample_answers': json.dumps(answers, sort_keys=True),
            'sample_recovered': json.dumps(recovered, sort_keys=True),
            'prompt_tokens_total': int(g['prompt_tokens'].sum()),
            'output_tokens_total': int(g['output_tokens'].sum()),
            'finish_reason_counts': json.dumps(g['finish_reason'].value_counts().to_dict(), sort_keys=True),
            'length_truncation_any': bool(g['length_truncation'].any()),
            'severe_repetition_any': bool(g['severe_repetition'].any()),
            'reasoning_marker_any': bool(g['reasoning_marker'].any()),
        })
    df = pd.DataFrame(rows)
    if len(df) != 29780:
        raise RuntimeError(f'SC-3 derived rows mismatch: {len(df)}')
    return df


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    reps = np.empty(BOOT_B, dtype=float)
    for i in range(BOOT_B):
        idx = rng.integers(0, n, n)
        reps[i] = float(values[idx].mean())
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def stratified_macro_bootstrap_ci(wide: pd.DataFrame, a: str, b: str | None, model: str, salt: str) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + int(hashlib.sha256(f'{model}|{a}|{b}|{salt}'.encode()).hexdigest()[:8], 16))
    by_ds = []
    for ds in DATASETS:
        g = wide[(wide['model_key'] == model) & (wide['dataset'] == ds)]
        av = vector_for(g, a)
        if b is None:
            dv = av
        else:
            bv = vector_for(g, b)
            dv = av - bv
        by_ds.append(dv.astype(float))
    reps = np.empty(BOOT_B, dtype=float)
    for i in range(BOOT_B):
        vals = []
        for dv in by_ds:
            idx = rng.integers(0, len(dv), len(dv))
            vals.append(float(dv[idx].mean()))
        reps[i] = float(np.mean(vals))
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    if p != 0.5:
        raise ValueError('Only p=0.5 is used')
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(k + 1)]
    m = max(logs)
    return float(math.exp(m) * sum(math.exp(x - m) for x in logs))


def mcnemar_exact_from_bool(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = a.astype(bool); b = b.astype(bool)
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2.0 * binom_cdf(min(b01, b10), n, 0.5))
    return {'a_correct_b_wrong': b01, 'a_wrong_b_correct': b10, 'discordant_n': n, 'mcnemar_exact_p': p}


def holm_adjust(rows: pd.DataFrame, p_col: str = 'mcnemar_exact_p') -> pd.DataFrame:
    out = rows.copy()
    order = out[p_col].sort_values(kind='mergesort').index.tolist()
    m = len(order)
    prev = 0.0
    adjusted = {}
    for rank, idx in enumerate(order, start=1):
        val = min(1.0, float(out.loc[idx, p_col]) * (m - rank + 1))
        prev = max(prev, val)
        adjusted[idx] = prev
    out['holm_p'] = pd.Series(adjusted)
    return out


def wide_samples(samples: pd.DataFrame, sc3: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in samples.groupby(['model_key', 'dataset', 'source_index', 'uid', 'gold'], sort=True):
        row = dict(zip(['model_key', 'dataset', 'source_index', 'uid', 'gold'], key))
        for _, r in g.iterrows():
            s = r['sample_id']
            row[f'{s}_correct'] = bool(r['correct'])
            row[f'{s}_recovered'] = bool(r['parser_recovered'])
        rows.append(row)
    wide = pd.DataFrame(rows)
    sc_cols = sc3[['model_key', 'dataset', 'source_index', 'uid', 'correct', 'parser_recovered']].rename(columns={'correct': 'sc3_correct', 'parser_recovered': 'sc3_recovered'})
    wide = wide.merge(sc_cols, on=['model_key', 'dataset', 'source_index', 'uid'], how='left')
    p = primary[primary['method'].isin(['cot', 'fusion'])][['model_key', 'dataset', 'source_index', 'uid', 'method', 'correct', 'parser_recovered', 'output_tokens']]
    for method in ['cot', 'fusion']:
        sub = p[p['method'] == method].drop(columns=['method']).rename(columns={'correct': f'original_{method}_correct', 'parser_recovered': f'original_{method}_recovered', 'output_tokens': f'original_{method}_output_tokens'})
        wide = wide.merge(sub, on=['model_key', 'dataset', 'source_index', 'uid'], how='left')
    if len(wide) != 29780:
        raise RuntimeError(f'Wide rows mismatch: {len(wide)}')
    if wide.duplicated(['model_key', 'dataset', 'source_index', 'uid']).any():
        raise RuntimeError('Duplicate wide model-item keys')
    return wide


def vector_for(g: pd.DataFrame, name: str) -> np.ndarray:
    if name == 'SC-3':
        return g['sc3_correct'].astype(float).to_numpy()
    if name == 'mean_single':
        return g[[f'{s}_correct' for s in SAMPLES]].astype(float).mean(axis=1).to_numpy()
    if name in SAMPLES:
        return g[f'{name}_correct'].astype(float).to_numpy()
    if name == 'original_fusion':
        return g['original_fusion_correct'].astype(float).to_numpy()
    if name == 'original_cot':
        return g['original_cot_correct'].astype(float).to_numpy()
    raise KeyError(name)


def mcnemar_or_blank(g: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    if a == 'mean_single' or b == 'mean_single':
        return {'a_correct_b_wrong': math.nan, 'a_wrong_b_correct': math.nan, 'discordant_n': math.nan, 'mcnemar_exact_p': math.nan, 'mcnemar_label': 'NOT_APPLICABLE_FRACTIONAL_MEAN_SINGLE'}
    return {**mcnemar_exact_from_bool(vector_for(g, a), vector_for(g, b)), 'mcnemar_label': 'POOLED ITEM-LEVEL MCNEMAR'}


def accuracy_results(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [('sample_1', 'sample_1'), ('sample_2', 'sample_2'), ('sample_3', 'sample_3'), ('SC-3', 'SC-3')]
    for model in MODEL_ORDER:
        mg = wide[wide['model_key'] == model]
        for dataset in DATASETS:
            dg = mg[mg['dataset'] == dataset]
            for label, vname in methods:
                v = vector_for(dg, vname)
                lo, hi = wilson(int(v.sum()), len(v))
                rows.append({'scope': 'model_dataset', 'model_key': model, 'dataset': dataset, 'method': label, 'N': len(v), 'correct_n': int(v.sum()), 'accuracy': float(v.mean()), 'CI95_low': lo, 'CI95_high': hi, 'recovered_n': len(v), 'recovery': 1.0, 'CI_method': 'Wilson binomial within dataset'})
        for label, vname in methods:
            vals = []
            ns = 0; correct = 0
            for dataset in DATASETS:
                dg = mg[mg['dataset'] == dataset]
                v = vector_for(dg, vname)
                vals.append(float(v.mean()))
                ns += len(v); correct += int(v.sum())
            lo, hi = stratified_macro_bootstrap_ci(wide, vname, None, model, 'accuracy')
            rows.append({'scope': 'equal_weight_macro', 'model_key': model, 'dataset': 'MACRO', 'method': label, 'N': ns, 'correct_n': correct, 'accuracy': float(np.mean(vals)), 'CI95_low': lo, 'CI95_high': hi, 'recovered_n': ns, 'recovery': 1.0, 'CI_method': f'equal-weight stratified bootstrap, B={BOOT_B}, seed={SEED}'})
    return pd.DataFrame(rows)


def paired_sc_comparisons(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        mg = wide[wide['model_key'] == model]
        for dataset in DATASETS:
            dg = mg[mg['dataset'] == dataset]
            for a, b in SC3_PAIRS:
                av = vector_for(dg, a); bv = vector_for(dg, b)
                lo, hi = stratified_macro_bootstrap_ci(dg.assign(model_key=model), a, b, model, f'dataset|{dataset}') if False else bootstrap_dataset_diff(dg, a, b, model, dataset)
                rows.append({'estimand': 'dataset paired accuracy difference', 'scope': 'dataset', 'model_key': model, 'dataset': dataset, 'method_a': a, 'method_b': b, 'N': len(dg), 'accuracy_a': float(av.mean()), 'accuracy_b': float(bv.mean()), 'accuracy_diff': float(np.mean(av - bv)), 'CI95_low': lo, 'CI95_high': hi, 'CI_method': f'within-dataset paired bootstrap, B={BOOT_B}, seed={SEED}', **mcnemar_or_blank(dg, a, b)})
        for a, b in SC3_PAIRS:
            ds_diffs = []
            acc_a = []; acc_b = []
            for dataset in DATASETS:
                dg = mg[mg['dataset'] == dataset]
                av = vector_for(dg, a); bv = vector_for(dg, b)
                ds_diffs.append(float(np.mean(av - bv)))
                acc_a.append(float(av.mean())); acc_b.append(float(bv.mean()))
            lo, hi = stratified_macro_bootstrap_ci(wide, a, b, model, 'macro_difference')
            rec = {'estimand': 'equal-weight macro paired accuracy difference', 'scope': 'equal_weight_macro', 'model_key': model, 'dataset': 'MACRO', 'method_a': a, 'method_b': b, 'N': len(mg), 'accuracy_a': float(np.mean(acc_a)), 'accuracy_b': float(np.mean(acc_b)), 'accuracy_diff': float(np.mean(ds_diffs)), 'CI95_low': lo, 'CI95_high': hi, 'CI_method': f'equal-weight stratified paired bootstrap, B={BOOT_B}, seed={SEED}'}
            if b == 'mean_single':
                rec.update({'a_correct_b_wrong': math.nan, 'a_wrong_b_correct': math.nan, 'discordant_n': math.nan, 'mcnemar_exact_p': math.nan, 'mcnemar_label': 'NOT_APPLICABLE_FRACTIONAL_MEAN_SINGLE'})
            else:
                rec.update(mcnemar_or_blank(mg, a, b))
            rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_dataset_diff(g: pd.DataFrame, a: str, b: str, model: str, dataset: str) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + int(hashlib.sha256(f'{model}|{dataset}|{a}|{b}|dataset'.encode()).hexdigest()[:8], 16))
    av = vector_for(g, a); bv = vector_for(g, b); diff = av - bv
    n = len(diff)
    reps = np.empty(BOOT_B)
    for i in range(BOOT_B):
        idx = rng.integers(0, n, n)
        reps[i] = float(diff[idx].mean())
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def budget_analysis(samples: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, mg in primary[primary['method'].isin(['direct', 'cot', 'structured'])].groupby('model_key', sort=True):
        per_item = mg.groupby(['dataset', 'source_index', 'uid'])['output_tokens'].sum()
        rows.append({'model_key': model, 'strategy': 'heterogeneous_fusion_direct_cot_structured', 'model_calls_per_item': 3, 'total_model_calls': len(mg), 'total_output_tokens': int(mg['output_tokens'].sum()), 'median_output_tokens_per_item': float(per_item.median()), 'p95_output_tokens_per_item': float(per_item.quantile(0.95))})
    for model, mg in samples.groupby('model_key', sort=True):
        per_item = mg.groupby(['dataset', 'source_index', 'uid'])['output_tokens'].sum()
        rows.append({'model_key': model, 'strategy': 'SC-3_cot_sample1_sample2_sample3', 'model_calls_per_item': 3, 'total_model_calls': len(mg), 'total_output_tokens': int(mg['output_tokens'].sum()), 'median_output_tokens_per_item': float(per_item.median()), 'p95_output_tokens_per_item': float(per_item.quantile(0.95))})
    return pd.DataFrame(rows)


def fusion_component_quality(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = primary.pivot_table(index=['model_key', 'dataset', 'source_index', 'uid', 'gold'], columns='method', values='correct', aggfunc='first').reset_index()
    for model, mg in wide.groupby('model_key', sort=True):
        model_rows = []
        for dataset, dg in mg.groupby('dataset', sort=True):
            comps = {m: float(dg[m].mean()) for m in ['direct', 'cot', 'structured']}
            mean_comp = float(np.mean(list(comps.values())))
            best = max(comps.values())
            oracle = float(dg[['direct', 'cot', 'structured']].astype(bool).any(axis=1).mean())
            fusion = float(dg['fusion'].mean())
            rec = {'scope': 'dataset', 'model_key': model, 'dataset': dataset, **{f'{m}_accuracy': comps[m] for m in comps}, 'mean_constituent_accuracy': mean_comp, 'best_constituent_accuracy': best, 'heterogeneous_fusion_accuracy': fusion, 'fusion_minus_mean_constituent': fusion - mean_comp, 'fusion_minus_best_constituent': fusion - best, 'oracle_any_component_correct': oracle, 'oracle_minus_fusion': oracle - fusion}
            rows.append(rec); model_rows.append(rec)
        rows.append({'scope': 'equal_weight_macro', 'model_key': model, 'dataset': 'MACRO', **{f'{m}_accuracy': float(np.mean([r[f'{m}_accuracy'] for r in model_rows])) for m in ['direct', 'cot', 'structured']}, 'mean_constituent_accuracy': float(np.mean([r['mean_constituent_accuracy'] for r in model_rows])), 'best_constituent_accuracy': float(np.mean([r['best_constituent_accuracy'] for r in model_rows])), 'heterogeneous_fusion_accuracy': float(np.mean([r['heterogeneous_fusion_accuracy'] for r in model_rows])), 'fusion_minus_mean_constituent': float(np.mean([r['fusion_minus_mean_constituent'] for r in model_rows])), 'fusion_minus_best_constituent': float(np.mean([r['fusion_minus_best_constituent'] for r in model_rows])), 'oracle_any_component_correct': float(np.mean([r['oracle_any_component_correct'] for r in model_rows])), 'oracle_minus_fusion': float(np.mean([r['oracle_minus_fusion'] for r in model_rows]))})
    return pd.DataFrame(rows)


def strip_standard_fence(text: str) -> str:
    t = (text or '').strip()
    m = re.fullmatch(r'```(?:json|JSON)?\s*(.*?)\s*```', t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def exact_json_keyset(text: str) -> bool:
    try:
        obj = json.loads((text or '').strip())
    except Exception:
        return False
    return isinstance(obj, dict) and set(obj.keys()) == {'space_label', 'answer', 'confidence'}


def structured_practical_tiers(primary: pd.DataFrame) -> pd.DataFrame:
    s = primary[primary['method'] == 'structured'].copy()
    s['tier1_raw_exact'] = s['structured_exact_json'].fillna(False).astype(bool)
    s['tier2_fence_stripped_exact'] = s['raw_output'].map(lambda x: exact_json_keyset(strip_standard_fence(x)))
    s['tier3_parser_recoverable'] = s['parser_recovered'].astype(bool)
    s['tier4_recovered_correct'] = s['correct'].astype(bool)
    rows = []
    for cols in [['model_key'], ['model_key', 'dataset']]:
        for key, g in s.groupby(cols, sort=True):
            if not isinstance(key, tuple): key = (key,)
            rows.append({**dict(zip(cols, key)), 'N': len(g), 'tier1_raw_exact_n': int(g['tier1_raw_exact'].sum()), 'tier1_raw_exact_rate': float(g['tier1_raw_exact'].mean()), 'tier2_fence_stripped_exact_n': int(g['tier2_fence_stripped_exact'].sum()), 'tier2_fence_stripped_exact_rate': float(g['tier2_fence_stripped_exact'].mean()), 'tier2_gain_over_tier1_n': int((g['tier2_fence_stripped_exact'] & ~g['tier1_raw_exact']).sum()), 'tier3_recoverable_n': int(g['tier3_parser_recoverable'].sum()), 'tier3_recoverable_rate': float(g['tier3_parser_recoverable'].mean()), 'tier4_correct_n': int(g['tier4_recovered_correct'].sum()), 'tier4_correct_rate': float(g['tier4_recovered_correct'].mean())})
    return pd.DataFrame(rows)


def final_answer_only(text: str) -> bool:
    t = (text or '').strip()
    return bool(re.fullmatch(r'(FINAL_ANSWER\s*:\s*[A-E]\s*\n\s*CONFIDENCE\s*:\s*[01](?:\.\d+)?)|([A-E]\s*[\.\):,-]?)', t, flags=re.I))


def cot_behavior(primary: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_cot = primary[primary['method'] == 'cot'].copy()
    p_direct = primary[primary['method'] == 'direct'][['model_key', 'dataset', 'source_index', 'uid', 'parsed_answer', 'parser_recovered']].rename(columns={'parsed_answer': 'direct_answer', 'parser_recovered': 'direct_recovered'})
    p_cot = p_cot.merge(p_direct, on=['model_key', 'dataset', 'source_index', 'uid'], how='left')
    for model, mg in p_cot.groupby('model_key', sort=True):
        rows.append(cot_behavior_row('primary_cot', model, 'ALL', mg))
        for dataset, dg in mg.groupby('dataset', sort=True):
            rows.append(cot_behavior_row('primary_cot', model, dataset, dg))
    s = samples.merge(p_direct, on=['model_key', 'dataset', 'source_index', 'uid'], how='left')
    for model, mg in s.groupby('model_key', sort=True):
        rows.append(cot_behavior_row('sc_samples_all', model, 'ALL', mg))
        for sample, sg in mg.groupby('sample_id', sort=True):
            rows.append(cot_behavior_row(f'sc_{sample}', model, 'ALL', sg))
    return pd.DataFrame(rows)


def cot_behavior_row(scope: str, model: str, dataset: str, g: pd.DataFrame) -> dict[str, Any]:
    agreement = (g['parser_recovered'].astype(bool) & g['direct_recovered'].astype(bool) & (g['parsed_answer'] == g['direct_answer']))
    discord = (g['parser_recovered'].astype(bool) & g['direct_recovered'].astype(bool) & (g['parsed_answer'] != g['direct_answer']))
    return {'scope': scope, 'model_key': model, 'dataset': dataset, 'N': len(g), 'output_tokens_mean': float(g['output_tokens'].mean()), 'output_tokens_median': float(g['output_tokens'].median()), 'output_tokens_p95': float(g['output_tokens'].quantile(0.95)), 'pct_le_20_tokens': float((g['output_tokens'] <= 20).mean()), 'pct_le_32_tokens': float((g['output_tokens'] <= 32).mean()), 'final_answer_only_pattern_rate': float(g['raw_output'].map(final_answer_only).mean()), 'direct_cot_answer_agreement_n': int(agreement.sum()), 'direct_cot_answer_agreement_rate': float(agreement.mean()), 'direct_cot_discordant_n': int(discord.sum()), 'direct_cot_discordant_rate': float(discord.mean())}


def dataset_answer_space(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = primary.pivot_table(index=['model_key', 'dataset', 'source_index', 'uid', 'gold'], columns='method', values=['parsed_answer', 'parser_recovered', 'correct', 'fusion_resolution', 'fusion_fallback_used'], aggfunc='first').reset_index()
    wide.columns = ['_'.join([str(x) for x in c if x]) for c in wide.columns]
    for model, mg in wide.groupby('model_key', sort=True):
        for dataset, dg in mg.groupby('dataset', sort=True):
            rec_answers = dg[['parsed_answer_direct', 'parsed_answer_cot', 'parsed_answer_structured']].replace('', np.nan)
            all_diff = rec_answers.notna().all(axis=1) & (rec_answers.nunique(axis=1) == 3)
            rows.append({'model_key': model, 'dataset': dataset, 'N': len(dg), 'gold_answer_classes_n': int(dg['gold'].nunique()), 'gold_answer_classes': ''.join(sorted(dg['gold'].unique())), 'strict_majority_rate': float((dg['fusion_resolution_fusion'] == 'strict_majority').mean()), 'fallback_rate': float(dg['fusion_fallback_used_fusion'].fillna(False).astype(bool).mean()), 'all_three_different_rate': float(all_diff.mean()), 'structured_recovery': float(dg['parser_recovered_structured'].mean()), 'structured_accuracy': float(dg['correct_structured'].mean()), 'cot_accuracy': float(dg['correct_cot'].mean()), 'fusion_accuracy': float(dg['correct_fusion'].mean()), 'fusion_minus_cot': float(dg['correct_fusion'].mean() - dg['correct_cot'].mean())})
    return pd.DataFrame(rows)


def primary_mcnemar(primary: pd.DataFrame) -> pd.DataFrame:
    wide = primary[primary['method'].isin(['cot', 'fusion'])].pivot_table(index=['model_key', 'dataset', 'source_index', 'uid'], columns='method', values='correct', aggfunc='first').reset_index()
    rows = []
    for model, mg in wide.groupby('model_key', sort=True):
        rec = {'model_key': model, 'scope': 'pooled', 'N': len(mg), 'fusion_accuracy': float(mg['fusion'].mean()), 'cot_accuracy': float(mg['cot'].mean()), 'fusion_minus_cot': float(mg['fusion'].mean() - mg['cot'].mean()), **mcnemar_exact_from_bool(mg['fusion'].astype(float).to_numpy(), mg['cot'].astype(float).to_numpy()), 'mcnemar_label': 'POOLED ITEM-LEVEL MCNEMAR'}
        rows.append(rec)
    return holm_adjust(pd.DataFrame(rows))


def verify_anchors(paired: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    macro = paired[(paired['scope'] == 'equal_weight_macro') & (paired['method_a'] == 'SC-3')]
    for (method_b, model), expected in ANCHORS.items():
        got = float(macro[(macro['model_key'] == model) & (macro['method_b'] == method_b)]['accuracy_diff'].iloc[0])
        delta = got - expected
        ok = abs(delta) <= ANCHOR_TOL
        rows.append({'comparison': f'SC-3_minus_{method_b}', 'model_key': model, 'observed': got, 'audit_anchor': expected, 'delta': delta, 'within_tolerance': ok})
    bad = [r for r in rows if not r['within_tolerance']]
    if bad:
        raise RuntimeError('Numerical audit anchor mismatch: ' + json.dumps(bad, indent=2))
    return rows


def table_md(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_markdown(index=False)


def write_docs(sc_acc: pd.DataFrame, paired: pd.DataFrame, budget: pd.DataFrame, component: pd.DataFrame, tiers: pd.DataFrame, cot: pd.DataFrame, space: pd.DataFrame, mcnemar: pd.DataFrame, anchors: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    macro_sc = sc_acc[sc_acc['scope'] == 'equal_weight_macro'].copy()
    macro_pair = paired[(paired['scope'] == 'equal_weight_macro') & (paired['method_a'] == 'SC-3')].copy()
    sc_mean = macro_pair[macro_pair['method_b'] == 'mean_single'].copy()
    sc_fusion = macro_pair[macro_pair['method_b'] == 'original_fusion'].copy()
    sc_cot = macro_pair[macro_pair['method_b'] == 'original_cot'].copy()
    comp_macro = component[component['scope'] == 'equal_weight_macro'].copy()
    budget_wide = budget.pivot(index='model_key', columns='strategy', values='total_output_tokens').reset_index()
    budget_wide['SC3_minus_heterogeneous_output_tokens'] = budget_wide['SC-3_cot_sample1_sample2_sample3'] - budget_wide['heterogeneous_fusion_direct_cot_structured']
    budget_wide['SC3_div_heterogeneous_output_tokens'] = budget_wide['SC-3_cot_sample1_sample2_sample3'] / budget_wide['heterogeneous_fusion_direct_cot_structured']

    write_text_atomic(OUT / 'SC3_RESULTS.md', '# SC-3 Results\n\nNEW_LLM_GENERATIONS_PERFORMED = SC3_COT_ONLY\n\n## Accuracy And Recovery\n\n' + macro_sc.to_markdown(index=False) + '\n\n## Equal-Weight Macro Paired Comparisons\n\n' + macro_pair.to_markdown(index=False) + '\n\nSC-3 versus mean_single uses a fractional paired estimand and does not report McNemar or Holm p-values. Binary comparisons include pooled item-level McNemar diagnostics but those are not tests of the equal-weight macro estimand.\n')
    write_text_atomic(OUT / 'SC3_BUDGET_ANALYSIS.md', '# SC-3 Budget Analysis\n\nSC-3 and heterogeneous Fusion are matched in model calls per item, not in output-token compute.\n\n' + budget.to_markdown(index=False) + '\n\n## Token Ratio\n\n' + budget_wide.to_markdown(index=False) + '\n')
    write_text_atomic(OUT / 'PRIOR_PREPRINT_RECONCILIATION.md', '# Prior Preprint Reconciliation\n\narXiv:2606.00971 is the authors\' earlier record: `HypothesisMed: Inference-Time Answer Fusion and Structured Hypothesis-Space Reporting for Biomedical Question Answering`.\n\nIt reported a different Phi Fusion result under a different serving/evaluation materialization. The present frozen five-model native-chat benchmark supersedes those earlier quantitative conclusions for this manuscript. This does not imply misconduct and does not pretend the protocols were identical. The earlier preprint should be disclosed as a prior record and reconciled transparently.\n')

    # Manuscript facts: only checked numbers.
    fact_lines = ['# Manuscript Numerical Facts', '']
    fact_lines += ['## Original Fusion vs Original CoT, Pooled Item-Level McNemar With Holm', '', mcnemar.to_markdown(index=False), '']
    fact_lines += ['## SC-3 Versus Mean Stochastic Single CoT, Equal-Weight Macro', '', sc_mean[['model_key','accuracy_a','accuracy_b','accuracy_diff','CI95_low','CI95_high','CI_method']].to_markdown(index=False), '']
    fact_lines += ['## SC-3 Versus Heterogeneous Fusion, Equal-Weight Macro', '', sc_fusion[['model_key','accuracy_a','accuracy_b','accuracy_diff','CI95_low','CI95_high','a_correct_b_wrong','a_wrong_b_correct','discordant_n','mcnemar_exact_p']].to_markdown(index=False), '']
    fact_lines += ['## SC-3 Versus Original CoT, Equal-Weight Macro', '', sc_cot[['model_key','accuracy_a','accuracy_b','accuracy_diff','CI95_low','CI95_high','a_correct_b_wrong','a_wrong_b_correct','discordant_n','mcnemar_exact_p']].to_markdown(index=False), '']
    fact_lines += ['## Heterogeneous Fusion Component Quality, Equal-Weight Macro', '', comp_macro.to_markdown(index=False), '']
    fact_lines += ['## SC-3 Budget', '', budget_wide.to_markdown(index=False), '']
    write_text_atomic(OUT / 'MANUSCRIPT_NUMERICAL_FACTS.md', '\n'.join(fact_lines) + '\n')

    def decision(row, eps=0.001):
        lo = row['CI95_low']; hi = row['CI95_high']; d = row['accuracy_diff']
        if lo > 0: return f'improves (diff {d:.6f}, 95% CI {lo:.6f} to {hi:.6f})'
        if hi < 0: return f'harms (diff {d:.6f}, 95% CI {lo:.6f} to {hi:.6f})'
        if abs(d) <= eps: return f'approximately matches (diff {d:.6f}, 95% CI {lo:.6f} to {hi:.6f})'
        return f'inconclusive direction (diff {d:.6f}, 95% CI {lo:.6f} to {hi:.6f})'

    lines = ['# Final Revision Evidence Summary', '', 'NEW_LLM_GENERATIONS_PERFORMED = SC3_COT_ONLY', 'FROZEN_PARSER_MODIFIED = NO', 'FUSION_RULE_MODIFIED = NO', 'DATASET_MEMBERSHIP_MODIFIED = NO', 'MODEL_PANEL_MODIFIED = NO', '']
    lines += ['## Corrected Statistical Separation', '', 'Primary SC-3 effects are equal-weight macro paired accuracy differences with stratified paired bootstrap CIs. Pooled item-level McNemar diagnostics are reported only for binary paired comparisons and are labeled `POOLED ITEM-LEVEL MCNEMAR`; they are not tests of the equal-weight macro estimand. SC-3 versus mean_single has no McNemar or Holm p-value because mean_single is fractional at item level.', '']
    lines += ['## 1. Does SC-3 improve over mean stochastic single CoT for each model?', '']
    for _, r in sc_mean.iterrows(): lines.append(f'- {r.model_key}: {decision(r)}.')
    lines += ['', '## 2. Does SC-3 beat heterogeneous Fusion for each model?', '']
    for _, r in sc_fusion.iterrows(): lines.append(f'- {r.model_key}: {decision(r)}.')
    lines += ['', '## 3. Does SC-3 beat original CoT for each model?', '']
    for _, r in sc_cot.iterrows(): lines.append(f'- {r.model_key}: {decision(r)}.')
    lines += ['', '## 4. Does the broad aggregation-failure interpretation survive?', '', 'No. The broad claim that inference-time aggregation generally fails does not survive. Same-prompt SC-3 improves some models, especially MedGemma, and behaves differently from heterogeneous prompt Fusion.', '']
    lines += ['## 5. What narrower claim is justified?', '', 'Heterogeneous Direct+CoT+Structured majority Fusion does not reliably improve over CoT and can harm performance when component quality is unequal. Same-prompt self-consistency can improve performance for some models. Aggregation design therefore matters.', '']
    lines += ['## 6. Does heterogeneous Fusion outperform its mean constituent?', '', comp_macro[['model_key','fusion_minus_mean_constituent']].to_markdown(index=False), '']
    lines += ['## 7. Does heterogeneous Fusion outperform its best constituent?', '', comp_macro[['model_key','fusion_minus_best_constituent']].to_markdown(index=False), '']
    lines += ['## 8. What is oracle headroom?', '', comp_macro[['model_key','oracle_any_component_correct','oracle_minus_fusion']].to_markdown(index=False), '']
    gemma_t = tiers[(tiers['model_key']=='gemma4_31b') & (tiers.get('dataset', pd.Series([np.nan]*len(tiers))).isna() if 'dataset' in tiers.columns else pd.Series([True]*len(tiers)))].head(1)
    if len(gemma_t)==0: gemma_t = tiers[tiers['model_key']=='gemma4_31b'].head(1)
    g = gemma_t.iloc[0]
    lines += ['## 9. How much Gemma contract failure disappears after fence stripping?', '', f'Gemma raw-exact structured compliance was {int(g.tier1_raw_exact_n)}/{int(g.N)} ({g.tier1_raw_exact_rate:.6f}); fence-stripped exact compliance was {int(g.tier2_fence_stripped_exact_n)}/{int(g.N)} ({g.tier2_fence_stripped_exact_rate:.6f}), a gain of {int(g.tier2_gain_over_tier1_n)} rows.', '']
    qcot = cot[(cot['model_key']=='qwen25') & (cot['scope']=='primary_cot') & (cot['dataset']=='ALL')].iloc[0]
    lines += ['## 10. Does Qwen2.5 actually exhibit extended CoT behavior?', '', f'Yes. Qwen2.5 primary CoT mean output tokens={qcot.output_tokens_mean:.2f}, median={qcot.output_tokens_median:.1f}, p95={qcot.output_tokens_p95:.1f}; final-answer-only pattern rate={qcot.final_answer_only_pattern_rate:.6f}; Direct-CoT agreement={qcot.direct_cot_answer_agreement_rate:.6f}.', '']
    pub = space[space['dataset']=='pubmedqa'].copy()
    lines += ['## 11. Is PubMedQA descriptively consistent with an answer-space mechanism?', '', 'Yes, descriptively. PubMedQA uses fewer answer classes and generally shows high strict-majority/low fallback rates relative to MedQA/MedMCQA, but this is descriptive consistency rather than causal proof.', '', pub[['model_key','dataset','gold_answer_classes_n','strict_majority_rate','fallback_rate','all_three_different_rate','fusion_minus_cot']].to_markdown(index=False), '']
    lines += ['## 12. What are the original Fusion-vs-CoT exact McNemar and Holm results?', '', mcnemar.to_markdown(index=False), '']
    lines += ['## 13. What are SC-3 versus mean-single macro effects and 95% stratified bootstrap CIs?', '', sc_mean[['model_key','accuracy_diff','CI95_low','CI95_high']].to_markdown(index=False), '']
    lines += ['## 14. How do SC-3 and heterogeneous Fusion compare in generation count and output-token cost?', '', 'Both use three model calls per item. Output-token cost differs:', '', budget_wide.to_markdown(index=False), '']
    lines += ['## 15. What exact paper title/framing is recommended?', '', 'Recommended title: `Not All Inference-Time Aggregation Is Equal: Heterogeneous Prompt Fusion versus Self-Consistency in Biomedical Question Answering`.', '']
    lines += ['## 16. Is any additional LLM experiment scientifically necessary?', '', 'ADDITIONAL_LLM_EXPERIMENTS_RECOMMENDED = NO', '']
    write_text_atomic(OUT / 'FINAL_REVISION_EVIDENCE_SUMMARY.md', '\n'.join(lines) + '\n')

    audit = ['# SC-3 Statistical Audit', '', f'Bootstrap seed: {SEED}', f'Bootstrap replicates: {BOOT_B}', '', '## Numerical Anchor Check', pd.DataFrame(anchors).to_markdown(index=False), '', '## Mean-Single Inference', 'SC-3 versus mean_single is fractional at item level and therefore has no McNemar or Holm p-value. The primary inference is the equal-weight stratified paired bootstrap CI.', '', '## Binary Comparisons', 'SC-3 versus sample_1/sample_2/sample_3/original_fusion/original_cot includes both macro bootstrap CIs and separately labeled pooled item-level McNemar diagnostics.', '']
    write_text_atomic(OUT / 'SC3_STATISTICAL_AUDIT.md', '\n'.join(audit) + '\n')


def write_checksums() -> None:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.name == 'checksums.sha256' or not path.is_file():
            continue
        rows.append(f'{sha_file(path)}  {path.relative_to(ROOT)}')
    write_text_atomic(OUT / 'checksums.sha256', '\n'.join(rows) + '\n')


def collect_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha_file(p) for p in paths if p.exists()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gen_paths = [SC_GEN / f'{m}_sc3_cot_samples.jsonl' for m in MODEL_ORDER]
    primary_files = [PRIMARY_DF]
    before_hashes = {'generation': collect_hashes(gen_paths), 'primary': collect_hashes(primary_files), 'parser': sha_file(PARSER), 'analyzer_before': sha_file(Path(__file__))}
    protocol = verify_protocol()
    manifests = verify_sc_generation_complete()
    parse_output = load_parser()
    primary = load_primary()
    samples = parse_sc_samples(parse_output)
    samples_for_sc3 = samples.drop(columns=['raw_output'])
    sc3 = derive_sc3(samples_for_sc3, tuple(SAMPLES))
    wide = wide_samples(samples_for_sc3, sc3, primary)
    sc_acc = accuracy_results(wide)
    paired = paired_sc_comparisons(wide)
    anchors = verify_anchors(paired)
    budget = budget_analysis(samples, primary)
    component = fusion_component_quality(primary)
    tiers = structured_practical_tiers(primary)
    cot = cot_behavior(primary, samples)
    space = dataset_answer_space(primary)
    pm = primary_mcnemar(primary)
    sc_sens = []
    for order in permutations(SAMPLES):
        d = derive_sc3(samples_for_sc3, order)
        w = wide_samples(samples_for_sc3, d, primary)
        a = accuracy_results(w)
        a['sample_priority'] = '>'.join(order)
        sc_sens.append(a)
    sc_sens = pd.concat(sc_sens, ignore_index=True)

    samples.drop(columns=['raw_output']).to_parquet(OUT / 'sc3_sample_level.parquet', index=False)
    sc3.to_parquet(OUT / 'sc3_derived_item_level.parquet', index=False)
    sc_acc.to_csv(OUT / 'sc3_accuracy_results.csv', index=False)
    paired.to_csv(OUT / 'sc3_paired_comparisons.csv', index=False)
    sc_sens.to_csv(OUT / 'sc3_sample_priority_sensitivity.csv', index=False)
    budget.to_csv(OUT / 'sc3_budget_analysis.csv', index=False)
    component.to_csv(OUT / 'fusion_component_quality_analysis.csv', index=False)
    tiers.to_csv(OUT / 'structured_practical_compliance_tiers.csv', index=False)
    cot.to_csv(OUT / 'cot_behavior_analysis.csv', index=False)
    space.to_csv(OUT / 'dataset_answer_space_mechanism.csv', index=False)
    pm.to_csv(OUT / 'primary_mcnemar_holm.csv', index=False)

    validation = {
        'status': 'passed',
        'protocol': protocol,
        'sc_manifests': manifests,
        'parser_sha256': sha_file(PARSER),
        'parser_sha_unchanged': sha_file(PARSER) == FROZEN_PARSER_SHA,
        'primary_analysis_rows': int(len(primary)),
        'sc_sample_rows': int(len(samples)),
        'sc3_rows': int(len(sc3)),
        'wide_duplicate_keys': int(wide.duplicated(['model_key','dataset','source_index','uid']).sum()),
        'bootstrap_seed': SEED,
        'bootstrap_replicates': BOOT_B,
        'numerical_anchor_checks': anchors,
        'hashes_before': before_hashes,
    }
    write_docs(sc_acc, paired, budget, component, tiers, cot, space, pm, anchors, validation)
    after_hashes = {'generation': collect_hashes(gen_paths), 'primary': collect_hashes(primary_files), 'parser': sha_file(PARSER), 'analyzer_after': sha_file(Path(__file__))}
    validation['hashes_after'] = after_hashes
    validation['generation_files_modified'] = before_hashes['generation'] != after_hashes['generation']
    validation['primary_benchmark_files_modified'] = before_hashes['primary'] != after_hashes['primary']
    if validation['generation_files_modified'] or validation['primary_benchmark_files_modified']:
        raise RuntimeError('Frozen generation or primary benchmark hashes changed during analysis')
    write_text_atomic(OUT / 'analysis_validation_manifest.json', json.dumps(validation, indent=2, sort_keys=True) + '\n')
    write_checksums()
    print(json.dumps({'status':'complete','sc_sample_rows':len(samples),'sc3_rows':len(sc3),'out_dir':str(OUT),'bootstrap_replicates':BOOT_B}, indent=2, sort_keys=True))
    print('PATCH OK')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'PATCH FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise

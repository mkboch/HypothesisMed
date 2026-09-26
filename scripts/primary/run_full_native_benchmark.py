#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList, set_seed

ROOT = Path(os.environ.get('HYPOTHESISMED_ROOT', Path(__file__).resolve().parents[2]))
PROTOCOL = ROOT / 'results/final_protocol_freeze_20260817'
OUT_DIR = ROOT / 'results/final_full_benchmark_20260817'
INPUTS = PROTOCOL / 'final_generation_inputs_native_chat.jsonl'
MAX_MODEL_LEN = 8192
MAX_TOKENS = 4096
SEED = 20260816
CONDITION = 'native_chat'
FSYNC_EVERY = 10

MODELS = {
    'qwen36': {'order': 1, 'model_id': 'Qwen/Qwen3.6-35B-A3B', 'revision': '995ad96eacd98c81ed38be0c5b274b04031597b0', 'snapshot': None, 'model_class': 'image_text_to_text', 'tokenizer_trust_remote_code': True, 'model_trust_remote_code': False, 'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_policy': 'explicit_non_reasoning_control', 'reasoning_control': 'apply_chat_template(enable_thinking=False)', 'decoding': {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5, 'repetition_penalty': 1.0}},
    'gemma4_31b': {'order': 2, 'model_id': 'google/gemma-4-31b-it', 'revision': '145dc2508c480a64b47242f160d286cff94a2343', 'snapshot': None, 'model_class': 'image_text_to_text', 'tokenizer_trust_remote_code': False, 'model_trust_remote_code': False, 'chat_template_kwargs': {}, 'reasoning_policy': 'native_output_observed', 'reasoning_control': 'native chat template; no explicit non-thinking control used', 'decoding': {'do_sample': True, 'temperature': 0.1, 'top_p': 1.0, 'top_k': None, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}},
    'medgemma27b': {'order': 3, 'model_id': 'google/medgemma-27b-it', 'revision': '2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f', 'snapshot': None, 'model_class': 'image_text_to_text', 'tokenizer_trust_remote_code': False, 'model_trust_remote_code': False, 'chat_template_kwargs': {}, 'reasoning_policy': 'native_reasoning_retained', 'reasoning_control': 'native model instruction serialization; no suppression or stripping; <unused94>thought retained verbatim', 'decoding': {'do_sample': False, 'temperature': None, 'top_p': None, 'top_k': None, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}},
    'qwen25': {'order': 4, 'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'revision': 'a09a35458c702b33eeacc393d103063234e8bc28', 'snapshot': None, 'model_class': 'causal_lm', 'tokenizer_trust_remote_code': True, 'model_trust_remote_code': False, 'chat_template_kwargs': {}, 'reasoning_policy': 'no_explicit_reasoning_mode', 'reasoning_control': 'native Qwen2.5 chat template; no thinking mode', 'decoding': {'do_sample': False, 'temperature': None, 'top_p': None, 'top_k': None, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}},
    'phi4mini': {'order': 5, 'model_id': 'microsoft/Phi-4-mini-instruct', 'revision': 'cfbefacb99257ffa30c83adab238a50856ac3083', 'snapshot': None, 'model_class': 'causal_lm', 'tokenizer_trust_remote_code': True, 'model_trust_remote_code': False, 'chat_template_kwargs': {}, 'reasoning_policy': 'no_explicit_reasoning_mode', 'reasoning_control': 'native Phi-4-mini instruction template; built-in Transformers Phi3ForCausalLM', 'decoding': {'do_sample': False, 'temperature': None, 'top_p': None, 'top_k': None, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}},
}

class PresencePenalty(LogitsProcessor):
    def __init__(self, penalty: float, prompt_lengths: list[int]):
        self.penalty = float(penalty); self.prompt_lengths = prompt_lengths
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.penalty == 0: return scores
        for row_idx in range(input_ids.shape[0]):
            generated = input_ids[row_idx, self.prompt_lengths[row_idx]:]
            if generated.numel(): scores[row_idx, torch.unique(generated)] -= self.penalty
        return scores

def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def severe_repetition(text: str) -> bool:
    words=re.findall(r'\S+', (text or '').lower())
    if len(words) < 4: return False
    grams=[tuple(words[i:i+4]) for i in range(len(words)-3)]
    counts=Counter(grams)
    return (1.0 - len(counts)/len(grams)) >= 0.50 or max(counts.values()) >= 20

def reasoning_marker(text: str) -> bool:
    t=(text or '').lower()
    return any(m in t for m in ['<think>','</think>','<|think|>','<|channel>thought','<|channel|>thought','<unused94>thought','<thought>','</thought>'])

def structured_exact_json(text: str):
    try: obj=json.loads((text or '').strip())
    except Exception: return False
    return isinstance(obj, dict) and set(obj.keys()) == {'space_label','answer','confidence'}

def render_native(tokenizer, prompt: str, kwargs: dict) -> str:
    k={'tokenize': False, 'add_generation_prompt': True}; k.update(kwargs)
    rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}], **k)
    if not isinstance(rendered, str): raise RuntimeError('chat template did not return string')
    return rendered

def load_completed(path: Path):
    done=set(); bad=0
    if not path.exists(): return done, bad
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip(): continue
            try: r=json.loads(line)
            except Exception:
                bad += 1; continue
            done.add(tuple(r['unique_key']))
    return done, bad

def generation_kwargs(cfg, eos_token_id, pad_token_id, prompt_length):
    d=cfg['decoding']
    kw={'max_new_tokens': MAX_TOKENS, 'do_sample': bool(d['do_sample']), 'eos_token_id': eos_token_id, 'pad_token_id': pad_token_id, 'repetition_penalty': d['repetition_penalty']}
    if d['do_sample']:
        kw.update(temperature=d['temperature'], top_p=d['top_p'])
        if d.get('top_k') is not None: kw['top_k']=d['top_k']
    if d.get('presence_penalty'):
        kw['logits_processor']=LogitsProcessorList([PresencePenalty(d['presence_penalty'], [prompt_length])])
    return kw

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-key', required=True, choices=sorted(MODELS)); args=ap.parse_args()
    cfg=MODELS[args.model_key]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path=OUT_DIR / f"{args.model_key}_native_chat_generations.jsonl"
    manifest_path=OUT_DIR / f"{args.model_key}_completion_manifest.json"
    log_path=OUT_DIR / f"{args.model_key}_run_status.json"
    inputs=load_jsonl(INPUTS)
    done,bad=load_completed(output_path)
    if bad: raise RuntimeError(f'{output_path} has {bad} corrupted JSONL rows; stop before resume')
    pending=[r for r in inputs if (cfg['model_id'], cfg['revision'], CONDITION, r['dataset'], int(r['source_index']), r['uid'], r['method'], SEED) not in done]
    print(f"MODEL={cfg['model_id']} REVISION={cfg['revision']} TOTAL={len(inputs)} DONE={len(done)} PENDING={len(pending)}", flush=True)
    log_path.write_text(json.dumps({'status':'loading','model_key':args.model_key,'done':len(done),'pending':len(pending),'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2)+'\n')
    if not pending:
        print('NO_PENDING_ROWS', flush=True); return 0
    set_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['revision'], local_files_only=False, trust_remote_code=cfg['tokenizer_trust_remote_code'])
    model_cls=AutoModelForImageTextToText if cfg['model_class']=='image_text_to_text' else AutoModelForCausalLM
    model=model_cls.from_pretrained(cfg['model_id'], revision=cfg['revision'], dtype=torch.bfloat16, local_files_only=False, trust_remote_code=cfg['model_trust_remote_code'])
    model.eval(); model.to('cuda:0')
    eos_token_id=model.generation_config.eos_token_id or getattr(model.config,'eos_token_id',None) or tokenizer.eos_token_id
    pad_token_id=tokenizer.pad_token_id
    if pad_token_id is None: pad_token_id=eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
    generated=0
    with output_path.open('a', encoding='utf-8') as f:
        for idx,row in enumerate(pending, start=1):
            serialized=render_native(tokenizer, row['prompt'], cfg['chat_template_kwargs'])
            enc=tokenizer(serialized, return_tensors='pt', return_dict=True)
            prompt_tokens=int(enc['input_ids'].shape[-1])
            if prompt_tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError(f"context overflow {row['dataset']} {row['source_index']} {row['method']} {prompt_tokens}+{MAX_TOKENS}>{MAX_MODEL_LEN}")
            enc=enc.to('cuda:0')
            with torch.inference_mode():
                out=model.generate(**enc, **generation_kwargs(cfg, eos_token_id, pad_token_id, prompt_tokens))
            gen_ids=out[0, enc['input_ids'].shape[-1]:]
            raw=tokenizer.decode(gen_ids, skip_special_tokens=True)
            diag=tokenizer.decode(gen_ids, skip_special_tokens=False)
            out_tokens=int(gen_ids.numel())
            rec={'unique_key':[cfg['model_id'], cfg['revision'], CONDITION, row['dataset'], int(row['source_index']), row['uid'], row['method'], SEED], 'model_key':args.model_key, 'model_id':cfg['model_id'], 'revision':cfg['revision'], 'backend':'transformers', 'runtime_environment':str(ROOT/'.venv_final'), 'python_version':platform.python_version(), 'torch_version':torch.__version__, 'cuda_build':torch.version.cuda, 'transformers_version':transformers.__version__, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'condition':CONDITION, 'method':row['method'], 'dataset':row['dataset'], 'source_index':int(row['source_index']), 'uid':row['uid'], 'gold':row.get('gold'), 'seed':SEED, 'decoding_settings':cfg['decoding'], 'reasoning_policy':cfg['reasoning_policy'], 'reasoning_control':cfg['reasoning_control'], 'max_model_len':MAX_MODEL_LEN, 'max_tokens':MAX_TOKENS, 'controlled_prompt':row['prompt'], 'controlled_prompt_sha256':row['controlled_prompt_sha256'], 'serialized_input':serialized, 'serialized_input_sha256':sha_text(serialized), 'prompt_tokens':prompt_tokens, 'raw_output':raw, 'diagnostic_decode_with_special_tokens':diag, 'output_tokens':out_tokens, 'finish_reason':'length' if out_tokens >= MAX_TOKENS else 'eos_or_stop', 'empty_completion':not bool(raw.strip()), 'length_truncation':out_tokens >= MAX_TOKENS, 'severe_repetition':severe_repetition(raw), 'reasoning_marker':reasoning_marker(diag), 'structured_exact_json':structured_exact_json(raw) if row['method']=='structured' else None, 'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n'); f.flush(); generated += 1
            if generated % FSYNC_EVERY == 0: os.fsync(f.fileno())
            if idx == 1 or idx % 25 == 0:
                print(f"row={idx}/{len(pending)} dataset={row['dataset']} method={row['method']} out={out_tokens} finish={rec['finish_reason']} empty={rec['empty_completion']} trunc={rec['length_truncation']} rep={rec['severe_repetition']} marker={rec['reasoning_marker']}", flush=True)
                log_path.write_text(json.dumps({'status':'running','model_key':args.model_key,'generated_this_run':generated,'pending_at_start':len(pending),'last_row':idx,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2)+'\n')
        os.fsync(f.fileno())
    # completion manifest, technical only
    records=load_jsonl(output_path)
    keys=[tuple(r['unique_key']) for r in records]
    manifest={'model_key':args.model_key,'model_id':cfg['model_id'],'revision':cfg['revision'],'condition':CONDITION,'expected_rows':len(inputs),'rows':len(records),'unique_keys':len(set(keys)),'duplicate_keys':len(keys)-len(set(keys)),'empty_completions':sum(r['empty_completion'] for r in records),'length_truncations':sum(r['length_truncation'] for r in records),'severe_repetition_rows':sum(r['severe_repetition'] for r in records),'reasoning_marker_rows':sum(r['reasoning_marker'] for r in records),'structured_exact_json':sum(r['structured_exact_json'] is True for r in records if r['method']=='structured'),'structured_rows':sum(r['method']=='structured' for r in records),'finish_reason_counts':dict(Counter(r['finish_reason'] for r in records)),'output_jsonl':str(output_path.relative_to(ROOT)),'output_sha256':sha_file(output_path),'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'complete':len(records)==len(inputs) and len(set(keys))==len(inputs)}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    log_path.write_text(json.dumps({'status':'complete','manifest':str(manifest_path.relative_to(ROOT)),'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2)+'\n')
    del model; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if manifest['complete'] else 1

if __name__ == '__main__':
    raise SystemExit(main())

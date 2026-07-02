#!/usr/bin/env python3
import json, os, shlex, sys
from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download, whoami

OUT = Path("/home/manikm/HypothesisMed/results/expanded_medgemma")
OUT.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit("HF_TOKEN missing")

print("HF_LOGIN_START", flush=True)
login(token=token, add_to_git_credential=True)
me = whoami(token=token)
print("HF_LOGIN_OK user=" + str(me.get("name", "unknown")), flush=True)

models = [
    ("MEDGEMMA_1_5_4B_IT", "medgemma_1_5_4b_it", "google/medgemma-1.5-4b-it", True),
    ("MEDGEMMA_27B_TEXT_IT", "medgemma_27b_text_it", "google/medgemma-27b-text-it", True),
    ("MEDGEMMA_4B_IT_LEGACY", "medgemma_4b_it_legacy", "google/medgemma-4b-it", False),
]

records = []
env_lines = []

for env_name, label, repo, main_target in models:
    print(f"\n=== ACCESS/DOWNLOAD {label} repo={repo} main_target={main_target} ===", flush=True)
    rec = {
        "env": env_name,
        "label": label,
        "repo": repo,
        "main_target": main_target,
        "ok": False,
        "path": "",
        "error": "",
    }
    try:
        HfApi().model_info(repo_id=repo, token=token)
        print(f"ACCESS_OK label={label}", flush=True)
        path = snapshot_download(repo_id=repo, token=token, local_files_only=False)
        rec["ok"] = True
        rec["path"] = path
        env_lines.append(f"{env_name}={shlex.quote(path)}")
        print(f"DOWNLOAD_OK label={label} path={path}", flush=True)
    except Exception as e:
        rec["error"] = repr(e)
        env_lines.append(f"{env_name}=''")
        print(f"DOWNLOAD_FAILED label={label} repo={repo} error={repr(e)[:1200]}", flush=True)
    records.append(rec)

(OUT / "medgemma_downloads.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
(OUT / "medgemma_model_paths.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

print("\n=== DOWNLOAD SUMMARY ===", flush=True)
for r in records:
    print(f"{r['label']}: ok={r['ok']} path={r['path']}", flush=True)

main_ok = [r for r in records if r["main_target"] and r["ok"]]
print(f"MAIN_TARGETS_OK={len(main_ok)}", flush=True)
print(f"WROTE={OUT / 'medgemma_model_paths.env'}", flush=True)

if len(main_ok) == 0:
    raise SystemExit("No main MedGemma target downloaded. Check that the token belongs to the account with accepted access.")

print("MEDGEMMA_DOWNLOAD_CHECK_OK", flush=True)

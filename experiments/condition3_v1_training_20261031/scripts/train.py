"""Train EXPERIMENTAL_CONDITION3_V1_CANDIDATE from the frozen training_template.yaml, or run a short smoke test.
usage: python train.py smoke|full   (only the template's path placeholders and run name are resolved)
Smoke differs only by epochs=1 and fraction=0.05 and is NOT a candidate model.
"""
import hashlib, json, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import torch, ultralytics, yaml
from ultralytics import YOLO

W = Path(r"C:\Users\ben\Documents\Codex\VERDE_CONDITION3_V1_TRAINING_20261031")
H = Path(r"C:\Users\ben\Documents\Codex\2026-10-24\files-pasted-by-the-user-verde\outputs\VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1")
BASE = H / "models" / "verde_condition3_best_validation.pt"
BASE_SHA = "54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987"
TEMPLATE_SHA = "fe7a8f9a093e119b4c58542bec6263b23790a805bcb20505ff59dd3dd05ae502"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FROZEN = dict(task="detect", mode="train", exist_ok=False, epochs=50, patience=10, imgsz=640, batch=4, device=0, workers=0,
              cache=False, amp=True, seed=20261023, deterministic=True, save=True, save_period=5, plots=True, resume=False)

mode = sys.argv[1]
assert mode in ("smoke", "full")
assert torch.cuda.is_available(), "CUDA required; refusing to train on CPU"
assert sha(BASE) == BASE_SHA, "baseline hash mismatch"
tpl = H / "configs" / "training_template.yaml"
assert sha(tpl) == TEMPLATE_SHA, "template hash mismatch"
cfg = yaml.safe_load(tpl.read_text(encoding="utf-8"))
assert all(cfg[k] == v for k, v in FROZEN.items()), "template differs from the frozen values"
name = "SMOKE_TEST_NOT_A_CANDIDATE" if mode == "smoke" else "EXPERIMENTAL_CONDITION3_V1_CANDIDATE"
cfg.update(model=str(BASE), data=str(W / "dataset" / "data.yaml"), project=str(W / "runs"), name=name)
assert not (W / "runs" / name).exists(), f"run {name} already exists; one controlled run only"
if mode == "smoke":
    cfg.update(epochs=1, fraction=0.05)  # shortcut: smoke trains on 5% for one epoch; never used as a model

rec = dict(mode=mode, resolved_config=cfg, git_commit="2f2849c7de94df0b7a416c14eca6b9c40456332e", argv=sys.argv, python=platform.python_version(),
           torch=torch.__version__, cuda_build=torch.version.cuda, ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0),
           vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), baseline_sha256=BASE_SHA, template_sha256=TEMPLATE_SHA,
           started_utc=datetime.now(timezone.utc).isoformat(),
           pip_freeze=subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines())
out = W / "config" / f"{mode}_run_record.json"
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")

kw = {k: v for k, v in cfg.items() if k not in ("model", "task", "mode")}
t0 = time.time()
YOLO(str(BASE)).train(**kw)
rec.update(finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=round(time.time() - t0, 1), peak_vram_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
run = W / "runs" / name
for ck in ("best.pt", "last.pt"):
    assert (run / "weights" / ck).exists(), ck
    rec[f"{ck}_sha256"] = sha(run / "weights" / ck)
assert sha(BASE) == BASE_SHA, "baseline changed during training"
rec["baseline_unchanged_after_run"] = True
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print("DONE", mode, rec["duration_s"], "s; peak VRAM", rec["peak_vram_gib"], "GiB")

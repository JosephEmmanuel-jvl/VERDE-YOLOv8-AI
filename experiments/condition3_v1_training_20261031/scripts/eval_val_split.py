"""Reporting-only: both models on the validation split at both sizes. Selects nothing; best.pt was already fixed by training."""
import json
from ultralytics import YOLO
W = "C:/Users/ben/Documents/Codex/VERDE_CONDITION3_V1_TRAINING_20261031"
M = {"baseline": "C:/Users/ben/Documents/Codex/2026-10-24/files-pasted-by-the-user-verde/outputs/VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1/models/verde_condition3_best_validation.pt",
     "candidate": W + "/runs/EXPERIMENTAL_CONDITION3_V1_CANDIDATE/weights/best.pt"}
N = {0: "Healthy", 1: "Yellowing", 2: "Wilting"}; out = {}
for sz in (640, 416):
    for t, p in M.items():
        b = YOLO(p).val(data=W + "/dataset/data.yaml", split="val", imgsz=sz, batch=4, conf=0.001, iou=0.7, device=0, workers=0, plots=False, verbose=False, project=W + "/eval/val_split", name=f"{t}_{sz}", exist_ok=True).box
        r = {N[int(c)]: dict(P=round(float(b.p[i]) * 100, 2), R=round(float(b.r[i]) * 100, 2), AP50=round(float(b.ap50[i]) * 100, 2), AP50_95=round(float(b.ap[i]) * 100, 2)) for i, c in enumerate(b.ap_class_index)}
        r["overall"] = dict(P=round(float(b.mp) * 100, 2), R=round(float(b.mr) * 100, 2), AP50=round(float(b.map50) * 100, 2), AP50_95=round(float(b.map) * 100, 2))
        out[f"{t}_{sz}"] = r; print(t, sz, r, flush=True)
json.dump(out, open(W + "/reports/validation_split_metrics.json", "w"), indent=1)

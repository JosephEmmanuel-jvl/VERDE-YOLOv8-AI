"""Evaluate one model on the historical test split, then run failure analysis and the two development checks.
usage: python evaluate.py TAG MODEL_PT IMGSZ   (identical settings for every model at a given IMGSZ; nothing is tuned)
Writes eval/TAG/{metrics.json,failures.json,sanity_12.json,post_v1_challenge.json} and annotated images.
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

W = Path(r"C:\Users\ben\Documents\Codex\VERDE_CONDITION3_V1_TRAINING_20261031")
tag, model_path, IMGSZ = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = W / "eval" / tag
(OUT / "failures").mkdir(parents=True, exist_ok=True)
NAMES = {0: "Healthy", 1: "Yellowing", 2: "Wilting"}
COL = {0: (0, 200, 0), 1: (255, 200, 0), 2: (255, 60, 200)}
VAL = dict(imgsz=IMGSZ, batch=4, conf=0.001, iou=0.7, device=0, workers=0)  # metric settings, same for every model
REPORT_CONF, MATCH_IOU, LOW_CONF = 0.25, 0.5, 0.5                         # failure-analysis settings, fixed before any result
model = YOLO(model_path)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def inside(gt, p):  # share of the GT box covered by prediction p
    ix = max(0, min(gt[2], p[2]) - max(gt[0], p[0])); iy = max(0, min(gt[3], p[3]) - max(gt[1], p[1]))
    return ix * iy / max((gt[2] - gt[0]) * (gt[3] - gt[1]), 1e-9)


def load_gt(lbl, w, h):
    rows = []
    for line in Path(lbl).read_text().strip().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        rows.append((int(c), [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return rows


def predict(path, conf=REPORT_CONF):
    r = model.predict(str(path), imgsz=VAL["imgsz"], conf=conf, iou=VAL["iou"], device=0, verbose=False)[0]
    return sorted(((int(c), float(s), b.tolist()) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)), key=lambda x: -x[1])


def draw(path, gts, preds, dest, title=""):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); d = ImageDraw.Draw(im); lw = max(2, im.width // 300)
    for c, b in gts: d.rectangle(b, outline=(255, 255, 255), width=lw)
    for c, s, b in preds:
        d.rectangle(b, outline=COL[c], width=lw); d.text((b[0] + 4, b[1] + 4), f"{NAMES[c]} {s:.2f}", fill=COL[c], font_size=max(12, im.width // 45))
    if title: d.text((6, 6), title, fill=(255, 255, 255), font_size=max(12, im.width // 50))
    im.thumbnail((1000, 1000)); im.save(dest, quality=90)


# 1. historical test metrics
m = model.val(data=str(W / "dataset" / "data.yaml"), split="test", project=str(OUT), name="val_test", exist_ok=True, plots=True, **VAL)
b = m.box; per = {}
for i, c in enumerate(b.ap_class_index):
    p, r = float(b.p[i]), float(b.r[i])
    per[NAMES[int(c)]] = dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), ap50=float(b.ap50[i]), ap50_95=float(b.ap[i]))
mp, mr = float(b.mp), float(b.mr)
per["overall"] = dict(precision=mp, recall=mr, f1=2 * mp * mr / max(mp + mr, 1e-9), ap50=float(b.map50), ap50_95=float(b.map))
cm = np.asarray(m.confusion_matrix.matrix).tolist()
json.dump(dict(tag=tag, model=model_path, settings=VAL, per_class=per, confusion_matrix_rows_pred_cols_true_with_background=cm), open(OUT / "metrics.json", "w"), indent=1)

# 2. failure analysis on the test split
D = W / "dataset"; cnt = Counter(); rows = []; fx = {c: Counter() for c in NAMES}
for ip in sorted((D / "images" / "test").iterdir()):
    w, h = ImageOps.exif_transpose(Image.open(ip)).size
    gts = load_gt(D / "labels" / "test" / (ip.stem + ".txt"), w, h); preds = predict(ip)
    for k in NAMES:  # historical definition: class-aware greedy matching at conf>=0.25, IoU>=0.5
        g = [x[1] for x in gts if x[0] == k]; free = set(range(len(g)))
        for c, s, pb in preds:
            if c != k: continue
            best = max(free, key=lambda j: iou(pb, g[j]), default=None)
            if best is not None and iou(pb, g[best]) >= MATCH_IOU: free.discard(best); fx[k]["tp"] += 1
            else: fx[k]["fp"] += 1
        fx[k]["fn"] += len(free)
    used, ev, confused = set(), [], set()
    for c, s, pb in preds:
        ious = [iou(pb, g[1]) for g in gts]; j = int(np.argmax(ious)) if ious else -1; best = ious[j] if ious else 0
        if len([g for g in gts if inside(g[1], pb) >= 0.7]) >= 2: ev.append(f"merged_plants:{NAMES[c]}")
        if best >= MATCH_IOU and gts[j][0] == c:
            if j in used: ev.append(f"duplicate:{NAMES[c]}")
            else:
                used.add(j)
                if c != 0 and s < LOW_CONF: ev.append(f"low_conf_symptom:{NAMES[c]}")
        elif best >= MATCH_IOU:
            ev.append(f"{NAMES[gts[j][0]]}_predicted_as_{NAMES[c]}"); confused.add(j)
        elif best >= 0.1: ev.append(f"poor_localization:{NAMES[c]}")
        else: ev.append(f"false_positive_background:{NAMES[c]}")
    for j, g in enumerate(gts):
        if j not in used and j not in confused: ev.append(f"missed:{NAMES[g[0]]}")
    cnt.update(ev); cnt["gt_boxes"] += len(gts); cnt["pred_boxes"] += len(preds); cnt["true_positives"] += len(used)
    if ev:
        rows.append(dict(image=ip.name, gt_classes=[NAMES[g[0]] for g in gts], events=ev, preds=[(NAMES[c], round(s, 3)) for c, s, _ in preds]))
        draw(ip, gts, preds, OUT / "failures" / (ip.stem + ".jpg"), ",".join(sorted(set(ev))))
def prf(t):
    p, r = t["tp"] / max(t["tp"] + t["fp"], 1), t["tp"] / max(t["tp"] + t["fn"], 1)
    return dict(tp=t["tp"], fp=t["fp"], fn=t["fn"], precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9))
fixed = {NAMES[k]: prf(v) for k, v in fx.items()}; fixed["overall"] = prf(sum(fx.values(), Counter()))
json.dump(fixed, open(OUT / "fixed_threshold_metrics.json", "w"), indent=1)
json.dump(dict(tag=tag, report_conf=REPORT_CONF, match_iou=MATCH_IOU, counts=dict(cnt), images_with_events=len(rows), rows=rows), open(OUT / "failures.json", "w"), indent=1)

# 3. TRAINING-DATA SANITY CHECK - NOT GENERALIZATION EVIDENCE
san = []
(OUT / "sanity_12").mkdir(exist_ok=True)
new = sorted(p for p in (D / "images" / "train").iterdir() if p.stem.startswith(("LETTUCEDEF_", "LAB_")))
assert len(new) == 12
for ip in new:
    w, h = ImageOps.exif_transpose(Image.open(ip)).size
    gts = load_gt(D / "labels" / "train" / (ip.stem + ".txt"), w, h); preds = predict(ip); want = gts[0][0]
    hit = sum(any(iou(p[2], g[1]) >= MATCH_IOU and p[0] == g[0] for p in preds) for g in gts)
    san.append(dict(image=ip.name, expected=NAMES[want], gt_boxes=len(gts), gt_found_correct_class=hit, pred_classes=dict(Counter(NAMES[p[0]] for p in preds)),
                    conf_min=round(min((p[1] for p in preds), default=0), 3), conf_mean=round(float(np.mean([p[1] for p in preds])) if preds else 0, 3)))
    draw(ip, gts, preds, OUT / "sanity_12" / (ip.stem + ".jpg"))
json.dump(dict(label="TRAINING-DATA SANITY CHECK - NOT GENERALIZATION EVIDENCE", rows=san), open(OUT / "sanity_12.json", "w"), indent=1)

# 4. POST-V1 DEVELOPMENT CHALLENGE (never used for training or tuning)
CH = json.load(open(r"C:\Users\ben\Documents\Codex\VERDE_YELLOWING_NEW_FAMILIES_20261029\annotations\FINAL_POST_V1_YELLOWING_ANNOTATIONS.json"))
assert len(CH) == 4
(OUT / "post_v1").mkdir(exist_ok=True); res = []
for a in CH:
    ip = Path(a["file"]); gt = [(1, a["bbox_xyxy"])]; preds = predict(ip, conf=0.05)
    res.append(dict(image=ip.name, approved_box=a["bbox_xyxy"], yellowing_detected_at_0_25=any(p[0] == 1 and p[1] >= REPORT_CONF and iou(p[2], a["bbox_xyxy"]) >= MATCH_IOU for p in preds),
                    preds_conf_ge_0_05=[dict(cls=NAMES[c], conf=round(s, 3), iou_with_approved_box=round(iou(bx, a["bbox_xyxy"]), 3)) for c, s, bx in preds]))
    draw(ip, gt, [p for p in preds if p[1] >= REPORT_CONF], OUT / "post_v1" / (ip.stem + ".jpg"))
json.dump(dict(label="POST-V1 DEVELOPMENT CHALLENGE - NOT INDEPENDENT ACCEPTANCE", rows=res), open(OUT / "post_v1_challenge.json", "w"), indent=1)
print("EVAL DONE", tag, json.dumps(per["overall"]), dict(cnt))

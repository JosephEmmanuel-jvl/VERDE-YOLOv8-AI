#!/usr/bin/env python3
"""Isolated middle-six condition fine-tune using every reviewed field crop.

Repeated views are intentionally retained. Unreviewed photos are inventoried,
but are not assigned guessed YOLO labels or empty-label negatives.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BASE = HERE / "condition_dataset"
OUT = HERE / "all_reviewed_middle_six_dataset_20261028_v2"
WEIGHTS = HERE / "runs/mounted_condition_v1/weights/best.pt"
PHONE = json.loads((HERE / "phone_batch_review/alignment.json").read_text(encoding="utf-8"))
OCT27 = json.loads((HERE / "oct27_wilt_alignment.json").read_text(encoding="utf-8"))
PHONE_LABELS = {"severe_wilt_left": 2, "small_wilt_right": 2,
                "healthy_large_left": 0, "healthy_frilly_right": 0}
NAMES = {0: "Healthy", 1: "Yellowing", 2: "Wilting"}


def add_crop(image, source, box, label, stem, records):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return False
    if min(x2 - x1, y2 - y1) < 80:
        return False
    crop = image[y1:y2, x1:x2]
    target = OUT / "images/train" / f"{stem}.jpg"
    if not cv2.imwrite(str(target), crop):
        raise OSError(target)
    (OUT / "labels/train" / f"{stem}.txt").write_text(
        f"{label} 0.5 0.5 0.94 0.94\n", encoding="utf-8")
    records.append({"source": source, "stem": stem, "label": NAMES[label],
                    "box_xyxy": [x1, y1, x2, y2]})
    return True


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUT}")
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            target = OUT / kind / split
            target.mkdir(parents=True)
            for source in (BASE / kind / split).iterdir():
                if source.is_file() and source.suffix.lower() in {".jpg", ".png", ".txt"}:
                    os.link(source, target / source.name)
    records = []
    coverage = {"phone_available": 192, "phone_reviewed": 0,
                "oct27_available": len(OCT27), "oct27_reviewed": 0,
                "center_six_available": 1, "center_six_reviewed": 1}

    for row in PHONE:
        if not row.get("accepted"):
            continue
        source = Path(row["path"])
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)
        used = False
        for name, label in PHONE_LABELS.items():
            box = row["boxes"][name]
            used |= add_crop(image, str(source), box, label,
                             f"phone-{row['id']:03d}-{name}", records)
        coverage["phone_reviewed"] += int(used)

    for row in OCT27:
        source = Path(row["path"])
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)
        used = False
        for name, key, accepted, label in (
            ("wilt", "box", "accepted", 2),
            ("healthy-control", "healthy_box", "healthy_accepted", 0),
        ):
            if row[accepted]:
                used |= add_crop(image, str(source), row[key], label,
                                 f"oct27-{row['number']}-{name}", records)
        coverage["oct27_reviewed"] += int(used)

    source = HERE / "pi-isolated-stage-20261028/center-six-crops/869316370_1658872398961593_2801133816909342676_n.jpg"
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(source)
    for slot, label, box in (
        (1, 0, (50, 190, 365, 500)),
        (2, 2, (485, 275, 675, 475)),
        (4, 0, (0, 565, 450, 965)),
        (5, 2, (445, 735, 685, 965)),
        (6, 0, (800, 580, 1325, 965)),
    ):
        if not add_crop(image, str(source), box, label,
                        f"oct28-center-slot{slot}", records):
            raise ValueError(f"Invalid center slot {slot}")

    (OUT / "data.yaml").write_text(
        f"path: '{OUT.as_posix()}'\ntrain: images/train\nval: images/val\n"
        "names:\n  0: Healthy\n  1: Yellowing\n  2: Wilting\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "description": "All eligible reviewed middle-view crops, including repeated photos",
        "inherited_train_crops": len(list((BASE / "images/train").glob("*.jpg")))
                                 + len(list((BASE / "images/train").glob("*.png"))),
        "new_crops": len(records), "coverage": coverage, "records": records,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"new_crops": len(records), "coverage": coverage}, indent=2), flush=True)
    return OUT / "data.yaml"


def main():
    yaml = prepare()
    model = YOLO(str(WEIGHTS))
    if model.names != NAMES:
        raise RuntimeError(f"Unexpected class map: {model.names}")
    model.train(data=str(yaml), project=str(HERE / "runs"),
                name="all_reviewed_middle_six_v2", exist_ok=False,
                epochs=8, patience=3, imgsz=640, batch=4, device=0,
                workers=0, optimizer="AdamW", lr0=0.000005, lrf=0.2,
                warmup_epochs=0, amp=False, mosaic=0, mixup=0, degrees=2,
                translate=0.02, scale=0.08, fliplr=0.5, flipud=0,
                seed=20261028, deterministic=True, plots=False, save=True)


if __name__ == "__main__":
    main()

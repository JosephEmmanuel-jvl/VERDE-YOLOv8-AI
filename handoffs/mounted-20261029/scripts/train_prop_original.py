"""Isolated v6 continuation with reviewed fixture-only negative tiles.

Sets 03-04 contain no printed props on lettuce. Sets 09-12 remain evaluation
scenes and are never copied into this training dataset.
"""
from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "pi-field-prop-candidate-20261026"
SOURCE = BASE / "dataset_pi_v6"
NEGATIVES = HERE / "prop_hard_negative_dataset"
OUT = HERE / "prop_v7_fixture_negative_dataset"
WEIGHTS = BASE / "runs/prop_detector_v6_pi_field/weights/best.pt"


def prepare() -> Path:
    if OUT.exists():
        raise FileExistsError(OUT)
    counts = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            dest = OUT / kind / split
            dest.mkdir(parents=True)
            for source in (SOURCE / kind / split).iterdir():
                if source.is_file() and source.suffix.lower() in (".jpg", ".png", ".txt"):
                    os.link(source, dest / source.name)
    for kind in ("images", "labels"):
        dest = OUT / kind / "train"
        files = sorted((NEGATIVES / kind / "train").glob("hard-negative-set0[34]-*"))
        counts[kind] = len(files)
        for source in files:
            if (dest / source.name).exists():
                raise FileExistsError(dest / source.name)
            os.link(source, dest / source.name)
    if counts != {"images": 32, "labels": 32}:
        raise RuntimeError(f"Unexpected negative count: {counts}")
    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: '{OUT.as_posix()}'\ntrain: images/train\nval: images/val\n"
        "names:\n  0: Printed Insect Prop\n", encoding="utf-8"
    )
    return yaml


if __name__ == "__main__":
    data = prepare()
    model = YOLO(str(WEIGHTS))
    if set(model.names.values()) != {"Printed Insect Prop"}:
        raise RuntimeError(model.names)
    model.train(
        data=str(data), project=str(HERE / "runs"),
        name="prop_v7_fixture_negatives", exist_ok=False,
        epochs=4, patience=2, imgsz=640, batch=4, device=0,
        workers=0, optimizer="AdamW", lr0=0.00001, lrf=0.2,
        warmup_epochs=0, amp=False, mosaic=0, mixup=0,
        degrees=8, translate=0.03, scale=0.1,
        fliplr=0.5, flipud=0.5, seed=20261028,
        deterministic=True, plots=False, save=True,
    )

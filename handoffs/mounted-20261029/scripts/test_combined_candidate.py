#!/usr/bin/env python3
"""Run the isolated four-model candidate on one saved camera JPEG.

This reads a saved frame and writes a separate result. It never opens the
camera or writes into VERDE's live backend directories.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PACKAGE = Path(__file__).resolve().parents[1]
LOCALIZER = PACKAGE / "weights/plant_localizer_reference.pt"
CONDITION = PACKAGE / "weights/condition_v2.pt"
HARVEST = PACKAGE / "weights/harvest_dedicated_experimental.pt"
PROP = PACKAGE / "weights/printed_prop_v7.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    points = list(range(0, length - tile + 1, stride))
    if points[-1] != length - tile:
        points.append(length - tile)
    return points


def predictions(result, names, wanted: set[str]) -> list[dict]:
    return sorted(({
        "class_name": names[int(box.cls.item())],
        "confidence": round(float(box.conf.item()), 4),
    } for box in result.boxes if names[int(box.cls.item())] in wanted),
        key=lambda item: item["confidence"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--localizer", type=Path, default=LOCALIZER)
    parser.add_argument("--localizer-imgsz", type=int, default=640)
    parser.add_argument("--exclude-regions", nargs="*", type=int, default=[],
                        help="Proposal region numbers outside the reviewed plant group")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--condition", type=Path, default=CONDITION)
    parser.add_argument("--harvest", type=Path, default=HARVEST)
    parser.add_argument("--prop", type=Path, default=PROP)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot decode {args.image}")
    height, width = image.shape[:2]
    localizer = YOLO(str(args.localizer))
    begun = time.monotonic()
    proposals = localizer.predict(image, imgsz=args.localizer_imgsz,
                                  conf=0.25, iou=0.5, device=args.device,
                                  verbose=False, save=False)[0]
    localizer_seconds = round(time.monotonic() - begun, 3)
    regions = []
    crops = []
    for number, box in enumerate(proposals.boxes, start=1):
        if number in args.exclude_regions:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(image[y1:y2, x1:x2])
        regions.append({"region": number, "box_xyxy": [x1, y1, x2, y2],
                        "localizer_confidence": round(float(box.conf.item()), 4)})

    times = {"localizer_seconds": localizer_seconds}
    del proposals, localizer
    release()
    condition = YOLO(str(args.condition))
    if set(condition.names.values()) != {"Healthy", "Yellowing", "Wilting"}:
        raise ValueError(f"Unexpected condition classes: {condition.names}")
    begun = time.monotonic()
    results = (condition.predict(crops, imgsz=640, conf=0.10, iou=0.5,
                                 batch=4, device=args.device, verbose=False,
                                 save=False) if crops else [])
    times["condition_seconds"] = round(time.monotonic() - begun, 3)
    for row, result in zip(regions, results):
        found = predictions(result, condition.names, set(condition.names.values()))
        row["condition"] = found[0] if found else None
        row["condition_candidates"] = found
    del results, condition
    release()

    harvest = YOLO(str(args.harvest))
    if "Harvest-Ready" not in harvest.names.values():
        raise ValueError(f"Harvest class missing: {harvest.names}")
    begun = time.monotonic()
    results = (harvest.predict(crops, imgsz=640, conf=0.10, iou=0.5,
                               batch=4, device=args.device, verbose=False,
                               save=False) if crops else [])
    times["harvest_seconds"] = round(time.monotonic() - begun, 3)
    for row, result in zip(regions, results):
        found = predictions(result, harvest.names, {"Harvest-Ready"})
        row["visual_harvest_candidate"] = found[0] if found else None
        row["visual_harvest_status"] = "UNVERIFIED"
    del results, harvest
    release()

    prop = YOLO(str(args.prop))
    if set(prop.names.values()) != {"Printed Insect Prop"}:
        raise ValueError(f"Unexpected prop classes: {prop.names}")
    tile, stride = 640, 384
    tiles, origins = [], []
    for top in starts(height, tile, stride):
        for left in starts(width, tile, stride):
            tiles.append(image[top:min(top + tile, height), left:min(left + tile, width)])
            origins.append((left, top))
    begun = time.monotonic()
    results = prop.predict(tiles, imgsz=640, conf=0.25, iou=0.5,
                           batch=4, device=args.device, verbose=False, save=False)
    times["prop_seconds"] = round(time.monotonic() - begun, 3)
    raw = []
    for (left, top), result in zip(origins, results):
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            raw.append({"box_xyxy": [x1 + left, y1 + top, x2 + left, y2 + top],
                        "confidence": round(float(box.conf.item()), 4)})
    del results, prop
    release()
    rectangles = [[int(round(p["box_xyxy"][0])), int(round(p["box_xyxy"][1])),
                   int(round(p["box_xyxy"][2] - p["box_xyxy"][0])),
                   int(round(p["box_xyxy"][3] - p["box_xyxy"][1]))] for p in raw]
    keep = cv2.dnn.NMSBoxes(rectangles, [p["confidence"] for p in raw],
                            0.25, 0.5) if raw else []
    props = [raw[int(index)] for index in keep]
    for prop_box in props:
        x1, y1, x2, y2 = prop_box["box_xyxy"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        prop_box["plant_region"] = next((r["region"] for r in regions
                                         if r["box_xyxy"][0] <= cx <= r["box_xyxy"][2]
                                         and r["box_xyxy"][1] <= cy <= r["box_xyxy"][3]), None)

    annotated = image.copy()
    for row in regions:
        x1, y1, x2, y2 = row["box_xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 220), 2)
        label = row["condition"]
        text = f'{row["region"]}: {label["class_name"]} {label["confidence"]:.2f}' if label else f'{row["region"]}: condition unknown'
        cv2.putText(annotated, text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 2)
        if row["visual_harvest_candidate"]:
            cv2.putText(annotated, f'Harvest? {row["visual_harvest_candidate"]["confidence"]:.2f}',
                        (x1, min(height - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 170, 0), 2)
    for item in props:
        x1, y1, x2, y2 = map(int, item["box_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 0, 200), 2)
        cv2.putText(annotated, f'PROP? {item["confidence"]:.2f}',
                    (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 0, 200), 2)

    args.output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(args.output / "combined-annotated.jpg"), annotated)
    report = {
        "scope": "Isolated saved-frame comparison; no app or Pi deployment",
        "source_image": str(args.image), "image_sha256": sha256(args.image),
        "models": {"localizer": {"path": str(args.localizer), "sha256": sha256(args.localizer)},
                   "condition": {"path": str(args.condition), "sha256": sha256(args.condition)},
                   "visual_harvest": {"path": str(args.harvest), "sha256": sha256(args.harvest)},
                   "printed_prop": {"path": str(args.prop), "sha256": sha256(args.prop)}},
        "plant_proposals": "Generated from this image by the packaged localizer",
        "excluded_regions": args.exclude_regions, "regions": regions,
        "simulated_pest_candidates": props, "prop_tiles": len(tiles),
        "raw_prop_candidates": len(raw), "times": times,
        "warning": "Harvest output has poor field separation; prop output may contain false alarms. Do not treat candidate boxes as validated app alerts.",
    }
    (args.output / "combined-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"regions": len(regions), "prop_candidates": len(props),
                      "times": times, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

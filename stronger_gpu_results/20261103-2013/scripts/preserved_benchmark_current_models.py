"""Benchmark current VERDE models on the isolated fixed-demo test groups."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, bw, bh = (float(value) for value in line.split())
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return boxes


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--prop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    condition = YOLO(str(args.condition))
    if list(condition.names.values()) != ["Healthy", "Yellowing", "Wilting"]:
        raise ValueError(condition.names)

    # Condition: one labelled plant crop per test image.
    condition_images = sorted((args.dataset / "condition_detect/images/test").glob("*.jpg"))
    condition_rows = []
    condition_confusion: Counter[tuple[str, str]] = Counter()
    per_condition = defaultdict(lambda: Counter(total=0, correct=0))
    for batch_start in range(0, len(condition_images), 4):
        batch_paths = condition_images[batch_start : batch_start + 4]
        predictions = condition.predict(
            batch_paths, imgsz=640, conf=0.10, iou=0.5, batch=4,
            device=args.device, verbose=False, save=False
        )
        for image_path, prediction in zip(batch_paths, predictions):
            label_path = args.dataset / "condition_detect/labels/test" / f"{image_path.stem}.txt"
            truth_index = int(label_path.read_text(encoding="utf-8").split()[0])
            truth = condition.names[truth_index]
            if len(prediction.boxes):
                best = max(prediction.boxes, key=lambda box: float(box.conf.item()))
                predicted = condition.names[int(best.cls.item())]
                confidence = float(best.conf.item())
            else:
                predicted, confidence = "No Detection", None
            correct = predicted == truth
            condition_confusion[(truth, predicted)] += 1
            per_condition[truth]["total"] += 1
            per_condition[truth]["correct"] += int(correct)
            condition_rows.append(
                {"image": image_path.name, "truth": truth,
                 "prediction": predicted, "confidence": confidence,
                 "correct": correct}
            )
        del predictions
    del condition
    gc.collect()
    torch.cuda.empty_cache()

    # Harvest classifier.
    harvest = YOLO(str(args.harvest))
    if list(harvest.names.values()) != ["not_ready", "ready"]:
        raise ValueError(harvest.names)
    harvest_images = sorted((args.dataset / "harvest_classify/test").glob("*/*.jpg"))
    harvest_rows = []
    harvest_confusion: Counter[tuple[str, str]] = Counter()
    for batch_start in range(0, len(harvest_images), 4):
        batch_paths = harvest_images[batch_start : batch_start + 4]
        predictions = harvest.predict(
            batch_paths, imgsz=640, batch=4, device=args.device,
            verbose=False, save=False
        )
        for image_path, prediction in zip(batch_paths, predictions):
            truth = image_path.parent.name
            scores = {
                harvest.names[index]: float(score)
                for index, score in enumerate(prediction.probs.data.tolist())
            }
            predicted = max(scores, key=scores.get)
            correct = predicted == truth
            harvest_confusion[(truth, predicted)] += 1
            harvest_rows.append(
                {"image": image_path.name, "truth": truth,
                 "prediction": predicted, "confidence": scores[predicted],
                 "ready_score": scores["ready"], "correct": correct}
            )
        del predictions
    del harvest
    gc.collect()
    torch.cuda.empty_cache()

    # Printed props: same tiled inference path used by the integrated worker.
    prop = YOLO(str(args.prop))
    if list(prop.names.values()) != ["Printed Insect Prop"]:
        raise ValueError(prop.names)
    prop_images = sorted((args.dataset / "prop_detect/images/test").glob("*.jpg"))
    prop_rows = []
    total_tp = total_fp = total_fn = 0
    for image_index, image_path in enumerate(prop_images, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(image_path)
        height, width = frame.shape[:2]
        tile, stride = 640, 384
        tiles, origins = [], []
        for top in tile_starts(height, tile, stride):
            for left in tile_starts(width, tile, stride):
                tiles.append(frame[top : top + tile, left : left + tile])
                origins.append((left, top))
        predictions = prop.predict(
            tiles,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            batch=8,
            device=args.device,
            verbose=False,
            save=False,
        )
        raw = []
        for (left, top), prediction in zip(origins, predictions):
            for box in prediction.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append(
                    {
                        "box": [x1 + left, y1 + top, x2 + left, y2 + top],
                        "confidence": float(box.conf.item()),
                    }
                )
        rectangles = [
            [
                int(round(item["box"][0])),
                int(round(item["box"][1])),
                int(round(item["box"][2] - item["box"][0])),
                int(round(item["box"][3] - item["box"][1])),
            ]
            for item in raw
        ]
        keep = (
            cv2.dnn.NMSBoxes(
                rectangles,
                [item["confidence"] for item in raw],
                0.25,
                0.45,
            )
            if raw
            else []
        )
        detections = [raw[int(index)] for index in keep]
        label_path = args.dataset / "prop_detect/labels/test" / f"{image_path.stem}.txt"
        ground_truth = load_yolo_boxes(label_path, width, height)
        candidates = sorted(
            (
                (iou(detection["box"], truth), detection_index, truth_index)
                for detection_index, detection in enumerate(detections)
                for truth_index, truth in enumerate(ground_truth)
            ),
            reverse=True,
        )
        matched_detections, matched_truth = set(), set()
        for overlap, detection_index, truth_index in candidates:
            if overlap < 0.5:
                break
            if detection_index in matched_detections or truth_index in matched_truth:
                continue
            matched_detections.add(detection_index)
            matched_truth.add(truth_index)
        tp = len(matched_detections)
        fp = len(detections) - tp
        fn = len(ground_truth) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        prop_rows.append(
            {
                "image": image_path.name,
                "truth_count": len(ground_truth),
                "detection_count": len(detections),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detections": detections,
            }
        )
        print(f"prop {image_index}/{len(prop_images)}: tp={tp} fp={fp} fn={fn}", flush=True)

    condition_correct = sum(row["correct"] for row in condition_rows)
    harvest_correct = sum(row["correct"] for row in harvest_rows)
    report = {
        "models": {
            "condition": str(args.condition),
            "harvest": str(args.harvest),
            "prop": str(args.prop),
        },
        "evaluation_scope": "Held-out fixed-demo capture groups only",
        "condition": {
            "images": len(condition_rows),
            "accuracy": safe_ratio(condition_correct, len(condition_rows)),
            "per_class_accuracy": {
                name: safe_ratio(counts["correct"], counts["total"])
                for name, counts in per_condition.items()
            },
            "confusion": {
                f"{truth}->{prediction}": count
                for (truth, prediction), count in sorted(condition_confusion.items())
            },
            "predictions": condition_rows,
        },
        "harvest": {
            "images": len(harvest_rows),
            "accuracy": safe_ratio(harvest_correct, len(harvest_rows)),
            "confusion": {
                f"{truth}->{prediction}": count
                for (truth, prediction), count in sorted(harvest_confusion.items())
            },
            "predictions": harvest_rows,
        },
        "prop": {
            "images": len(prop_rows),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": safe_ratio(total_tp, total_tp + total_fp),
            "recall": safe_ratio(total_tp, total_tp + total_fn),
            "f1": safe_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
            "predictions": prop_rows,
        },
        "limitations": [
            "Same mounted camera and same plants; not a generalization estimate",
            "Condition and harvest truth is repeated across near-duplicate frames",
            "Printed props are presentation surrogates, not biological pests",
        ],
    }
    (args.output / "baseline-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    compact = {
        "condition": {key: value for key, value in report["condition"].items() if key != "predictions"},
        "harvest": {key: value for key, value in report["harvest"].items() if key != "predictions"},
        "prop": {key: value for key, value in report["prop"].items() if key != "predictions"},
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

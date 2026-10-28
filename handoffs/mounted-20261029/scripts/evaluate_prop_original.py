"""Compare preserved v6 and isolated v7 on reserved Pi camera scenes."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from evaluate_props import GT_PREVIEW, iou, scan


HERE = Path(__file__).resolve().parent
PHOTOS = Path(r"C:\Users\josephemmanuel\Desktop\VERDE AI\Pi-camera-field-20261027-48")
WEIGHTS = {
    "preserved_v6": HERE.parent / "pi-field-prop-candidate-20261026/runs/prop_detector_v6_pi_field/weights/best.pt",
    "fixture_negative_v7": HERE / "runs/prop_v7_fixture_negatives/weights/best.pt",
}


def main() -> None:
    rows = []
    for name, weight in WEIGHTS.items():
        model = YOLO(str(weight))
        for set_no in (9, 10, 11, 12):
            photo = sorted((PHOTOS / f"set-{set_no:02d}").glob("*.jpg"))[0]
            image = cv2.imread(str(photo))
            if image is None:
                raise FileNotFoundError(photo)
            gt = [[round(v * 1.125) for v in box] for box in GT_PREVIEW.get(set_no, [])]
            predictions = scan(model, image)
            matched = set()
            matched_scores = []
            for pred in sorted(predictions, key=lambda x: -x["confidence"]):
                choices = [(iou(pred["xyxy"], box), i) for i, box in enumerate(gt)
                           if i not in matched]
                if choices and max(choices)[0] >= 0.3:
                    matched.add(max(choices)[1])
                    matched_scores.append(pred["confidence"])
            row = {
                "model": name, "set": set_no, "source": photo.name,
                "reviewed_on_leaf": len(gt), "matched": len(matched),
                "missed": len(gt) - len(matched),
                "extra_boxes": len(predictions) - len(matched),
                "matched_confidences": matched_scores,
                "predictions": predictions,
            }
            rows.append(row)
            print(name, set_no, f"matched {len(matched)}/{len(gt)}",
                  f"extra {row['extra_boxes']}", flush=True)
    (HERE / "prop_v7_fixture_negative_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

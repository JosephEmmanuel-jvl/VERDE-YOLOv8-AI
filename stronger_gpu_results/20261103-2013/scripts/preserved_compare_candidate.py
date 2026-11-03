"""Compare preserved and adapted condition weights on untouched images."""

import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / "work/pi-test-package-20261029/weights/condition_v2.pt"
CANDIDATE = HERE / "runs/condition_v2_yellowing_adaptation/weights/best.pt"
DATA = HERE / "candidate_dataset"
EXPECTED_BASELINE = "36c4155af38af0e9eaac4cf8c487b24d6a6790f57c3f144a71c69908b0f0269c"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_heldout(model):
    records = []
    for photo in sorted((DATA / "heldout").glob("YEL-*.png")):
        result = model.predict(str(photo), imgsz=640, conf=0.10,
                               iou=0.50, device=0, verbose=False, save=False)[0]
        detections = sorted(({
            "class": model.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 6),
            "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
        } for box in result.boxes), key=lambda item: item["confidence"], reverse=True)
        records.append({"image": photo.name, "top": detections[0] if detections else None,
                        "yellowing_max_confidence": max(
                            (item["confidence"] for item in detections if item["class"] == "Yellowing"),
                            default=None), "detections": detections})
    return records


def test_original_validation(model):
    metrics = model.val(data=str(DATA / "data.yaml"), split="val", imgsz=640,
                        conf=0.001, iou=0.50, device=0, batch=4, workers=0,
                        verbose=False, plots=False, project=str(HERE / "comparisons"),
                        name="validation", exist_ok=True)
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {
            model.names[int(class_id)]: {
                "ap50": float(metrics.box.ap50[index]),
                "ap50_95": float(metrics.box.all_ap[index].mean()),
            }
            for index, class_id in enumerate(metrics.box.ap_class_index)
        },
    }


def main():
    if sha256(BASELINE) != EXPECTED_BASELINE:
        raise RuntimeError("Protected Condition V2 hash mismatch")
    if not CANDIDATE.is_file():
        raise FileNotFoundError(CANDIDATE)
    report = {"limitations": [
        "Outside Yellowing images are from the same source collection; plant-level independence is unconfirmed.",
        "Only 12 held-out outside photos and the original validation set are compared.",
        "These results do not establish fresh Pi-camera or deployed-app accuracy."],
              "models": {}}
    for name, path in (("baseline", BASELINE), ("candidate", CANDIDATE)):
        print(f"Comparing {name}: {path}", flush=True)
        model = YOLO(str(path))
        if list(model.names.values()) != ["Healthy", "Yellowing", "Wilting"]:
            raise RuntimeError(f"Unexpected classes in {path}: {model.names}")
        report["models"][name] = {
            "path": str(path), "sha256": sha256(path), "classes": model.names,
            "heldout": test_heldout(model),
            "original_validation": test_original_validation(model),
        }
        (HERE / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if sha256(BASELINE) != EXPECTED_BASELINE:
        raise RuntimeError("Protected Condition V2 changed")
    print("Comparison saved:", HERE / "comparison.json", flush=True)


if __name__ == "__main__":
    main()

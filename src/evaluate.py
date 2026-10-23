"""Evaluate a VERDE three-class checkpoint and save machine-readable metrics."""

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


EXPECTED_NAMES = {0: "Healthy", 1: "Yellowing", 2: "Wilting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data = args.data.resolve()
    model = YOLO(str(model_path))
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_NAMES:
        raise ValueError(f"Unexpected class mapping: {names}")

    metrics = model.val(
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, plots=True, project="runs/evaluation",
        name=model_path.stem, exist_ok=False,
    )
    report = {
        "model": str(model_path),
        "sha256": sha256(model_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "ap50_95_by_class": {
            names[index]: float(value) for index, value in enumerate(metrics.box.maps)
        },
    }
    output = (args.output or Path("runs/evaluation") / f"{model_path.stem}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()


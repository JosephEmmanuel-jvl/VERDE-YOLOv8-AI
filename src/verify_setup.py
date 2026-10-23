"""Verify model integrity, CUDA availability, and prepared dataset counts."""

import argparse
import hashlib
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "verde_condition3_best_validation.pt": "54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987",
    "verde_condition3_final_gpu_resume.pt": "d98118878dfb3b6fdd0bb5b71237f8bbdbbeb812c4acfc6b338e1a31cb049bf5",
}
EXPECTED_NAMES = {0: "Healthy", 1: "Yellowing", 2: "Wilting"}
EXPECTED_COUNTS = {"train": 760, "val": 207, "test": 200}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename, expected_hash in EXPECTED.items():
        path = ROOT / "models" / filename
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {filename}: {actual_hash}")
        names = {int(key): str(value) for key, value in YOLO(str(path)).names.items()}
        if names != EXPECTED_NAMES:
            raise ValueError(f"Unexpected classes in {filename}: {names}")
        print(f"model OK: {filename}")

    data_path = args.data.resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    for split, expected_count in EXPECTED_COUNTS.items():
        label_dir = dataset_root / "labels" / split
        count = len(list(label_dir.glob("*.txt")))
        if count != expected_count:
            raise ValueError(f"{split}: expected {expected_count} labels, found {count}")
        print(f"dataset {split}: {count}")

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()

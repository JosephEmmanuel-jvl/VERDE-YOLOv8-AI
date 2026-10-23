"""Start a new isolated VERDE three-class YOLO training run."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    model_path = args.model.resolve()
    if not data.is_file() or not model_path.is_file():
        raise FileNotFoundError("The data YAML and starting checkpoint are required")
    if str(args.device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build")

    YOLO(str(model_path)).train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False,
        amp=True,
        seed=20261023,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        save_period=5,
        plots=True,
    )


if __name__ == "__main__":
    main()


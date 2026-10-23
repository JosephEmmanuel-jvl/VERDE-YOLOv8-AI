"""Build an isolated three-class dataset from the unchanged VERDE V2 split."""

import argparse
import csv
import os
import shutil
from pathlib import Path


NAMES = {0: "Healthy", 1: "Yellowing", 2: "Wilting"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    return parser.parse_args()


def find_image(source: Path, decoded: Path, split: str, stem: str) -> Path:
    matches = [
        path
        for path in (source / "images" / split).glob(stem + ".*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source image for {split}/{stem}: {matches}")
    image = matches[0]
    if image.suffix.lower() == ".heic":
        image = decoded / split / (stem + ".jpg")
        if not image.is_file():
            raise FileNotFoundError(image)
    return image


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    decoded = args.decoded.resolve()
    dest = args.dest.resolve()
    if not source.is_dir() or not decoded.is_dir():
        raise FileNotFoundError("Both VERDE source folders are required")
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {dest}")

    records = []
    for split in ("train", "val", "test"):
        for label in sorted((source / "labels" / split).glob("*.txt")):
            rows = [
                line.split()
                for line in label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                continue
            if any(len(row) != 5 for row in rows):
                raise ValueError(f"Unexpected label format: {label}")
            class_ids = {int(row[0]) for row in rows}
            if class_ids - NAMES.keys():
                continue

            image = find_image(source, decoded, split, label.stem)
            out_image = dest / "images" / split / (label.stem + image.suffix.lower())
            out_label = dest / "labels" / split / label.name
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            transfer = link_or_copy(image, out_image)
            shutil.copy2(label, out_label)
            records.append((split, label.stem, image.suffix.lower(), transfer,
                            ";".join(map(str, sorted(class_ids)))))

    with (dest / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "stem", "suffix", "transfer", "classes"))
        writer.writerows(records)

    (dest / "data.yaml").write_text(
        f"path: {dest.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: Healthy\n  1: Yellowing\n  2: Wilting\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(row[0] == split for row in records)}")
    print(dest / "data.yaml")


if __name__ == "__main__":
    main()


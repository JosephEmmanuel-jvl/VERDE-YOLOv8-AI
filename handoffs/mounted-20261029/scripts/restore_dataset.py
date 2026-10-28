"""Verify and restore a reviewed YOLO dataset package on the training machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    dest = args.destination.resolve()
    manifest = json.loads((package / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    archive_path = package / "_verified_restore.zip"
    if archive_path.exists():
        raise FileExistsError(archive_path)
    try:
        with archive_path.open("wb") as output:
            for part in manifest["parts"]:
                path = package / part["name"]
                with path.open("rb") as input_file:
                    if digest(input_file) != part["sha256"]:
                        raise ValueError(f"Hash mismatch: {path}")
                with path.open("rb") as input_file:
                    for block in iter(lambda: input_file.read(1024 * 1024), b""):
                        output.write(block)
        with archive_path.open("rb") as archive_file:
            if digest(archive_file) != manifest["archive_sha256"]:
                raise ValueError("Reconstructed archive hash mismatch")
        with zipfile.ZipFile(archive_path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failed: {bad}")
            root = dest.resolve()
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")
            dest.mkdir(parents=True)
            archive.extractall(dest)
        yaml = dest / "data.yaml"
        lines = yaml.read_text(encoding="utf-8").splitlines()
        lines = [f"path: '{dest.as_posix()}'" if line.startswith("path:") else line for line in lines]
        yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"VERIFIED DATASET: {dest} ({manifest['source_files']} files)")
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

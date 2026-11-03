#!/usr/bin/env python3
"""Verify, reconstruct, and safely extract the VERDE continuation savepoint."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


PARTS = {
    "VERDE-field-adaptation-and-training-continuation-20261103.zip.part001":
        "ca5863b556fc13277a068802382cc06f5be6e9242beb1b12c3ed579c62494f4a",
    "VERDE-field-adaptation-and-training-continuation-20261103.zip.part002":
        "78f71670b61a2fcb2e67b8f47969f9b9ec95a017c9d3d81663b6dd6cafcb7a71",
}
ZIP_SHA256 = "c04c23025cd9d10e7dc9f9a3bb3dcc96ca2e2b8238940377c04e22b5dc0a21f1"
ZIP_NAME = "VERDE-field-adaptation-and-training-continuation-20261103.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile, destination: Path):
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe ZIP path: {member.filename}")
        yield member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty: {args.output_dir}")

    for name, expected in PARTS.items():
        part = args.parts_dir / name
        if not part.is_file():
            raise FileNotFoundError(part)
        actual = sha256(part)
        if actual != expected:
            raise ValueError(f"Part checksum mismatch: {name}")
        print(f"Verified {name}")

    with tempfile.TemporaryDirectory(prefix="verde-continuation-") as temporary:
        archive_path = Path(temporary) / ZIP_NAME
        with archive_path.open("wb") as output:
            for name in PARTS:
                with (args.parts_dir / name).open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)

        if sha256(archive_path) != ZIP_SHA256:
            raise ValueError("Reconstructed ZIP checksum mismatch")
        print("Verified reconstructed ZIP")

        with zipfile.ZipFile(archive_path) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"Corrupt ZIP member: {corrupt}")
            archive.extractall(args.output_dir, members=safe_members(archive, args.output_dir))

    print(f"Restored savepoint to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

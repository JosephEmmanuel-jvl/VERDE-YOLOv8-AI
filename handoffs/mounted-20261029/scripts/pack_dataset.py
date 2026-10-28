"""Archive an existing reviewed YOLO dataset into verified Git LFS parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PART_SIZE = 800 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    zip_path = output / f"{source.name}.zip"
    if zip_path.exists():
        raise FileExistsError(zip_path)
    files = sorted(p for p in source.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
        for path in files:
            archive.write(path, path.relative_to(source).as_posix(), compress_type=zipfile.ZIP_STORED)
    parts = []
    with zip_path.open("rb") as inp:
        for index, first in enumerate(iter(lambda: inp.read(PART_SIZE), b""), 1):
            part = output / f"{zip_path.name}.part{index:03d}"
            if part.exists():
                raise FileExistsError(part)
            part.write_bytes(first)
            parts.append({"name": part.name, "bytes": part.stat().st_size, "sha256": digest(part)})
    manifest = {"source_name": source.name, "source_files": len(files),
                "archive_bytes": zip_path.stat().st_size, "archive_sha256": digest(zip_path),
                "parts": parts}
    (output / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    # The complete ZIP is a temporary packaging artifact; the verified parts are the deliverable.
    zip_path.unlink()


if __name__ == "__main__":
    main()

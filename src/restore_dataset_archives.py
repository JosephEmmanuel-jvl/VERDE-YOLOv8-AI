"""Verify, rebuild, and extract the VERDE dataset from Git LFS parts."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


ARCHIVES = ("verde_v2.zip", "verde_v2_anylabeling.zip")
BUFFER_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        checksums[name.strip()] = expected.lower()
    return checksums


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        bad_member = zip_file.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member in {archive.name}: {bad_member}")
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"Unsafe ZIP path: {member.filename}")
        zip_file.extractall(destination)


def restore_archive(
    archive_name: str,
    source_dir: Path,
    archive_dir: Path,
    destination: Path,
    checksums: dict[str, str],
) -> None:
    parts = sorted(source_dir.glob(f"{archive_name}.part*"))
    if not parts:
        raise FileNotFoundError(f"No Git LFS parts found for {archive_name}")

    for part in parts:
        if part.name not in checksums:
            raise RuntimeError(f"Missing checksum entry for {part.name}")
        print(f"Verifying {part.name}...")
        verify(part, checksums[part.name])

    archive_path = archive_dir / archive_name
    print(f"Joining {len(parts)} parts into {archive_path}...")
    with archive_path.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=BUFFER_SIZE)

    verify(archive_path, checksums[archive_name])
    print(f"Extracting {archive_name}...")
    safe_extract(archive_path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Folder that will receive verde_v2 and verde_v2_anylabeling",
    )
    parser.add_argument(
        "--remove-archives",
        action="store_true",
        help="Delete reconstructed ZIP files after successful extraction",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    source_dir = repository / "dataset_sources"
    checksums = load_checksums(source_dir / "SHA256SUMS.txt")

    destination = args.dest.resolve()
    archive_dir = destination / "_archives"
    destination.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for archive_name in ARCHIVES:
        restore_archive(
            archive_name, source_dir, archive_dir, destination, checksums
        )

    if args.remove_archives:
        for archive_name in ARCHIVES:
            (archive_dir / archive_name).unlink()
        archive_dir.rmdir()

    print("Dataset archives verified and extracted successfully.")


if __name__ == "__main__":
    main()

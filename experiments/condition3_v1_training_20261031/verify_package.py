"""Self-check for this package. Exits non-zero on any failure.
usage: python verify_package.py [--load]     (--load also opens best.pt with Ultralytics)
"""
import csv, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEST = "e6b3182c9edc67e51a65d660c0566122b85f2d0a9d173f6a133c81d55d40ffe9"
LAST = "1b9142fd118e57f3bff6e77ecc418d5b6a73ceadad3fe5ac7ce170a2e4fba5d8"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

n = 0
for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    digest, rel = line.split("  ", 1)
    assert sha(ROOT / rel) == digest, f"checksum mismatch: {rel}"
    n += 1
assert n > 200, n

assert sha(ROOT / "weights/best.pt") == BEST, "best.pt hash"
assert sha(ROOT / "weights/last.pt") == LAST, "last.pt hash"

rows = list(csv.DictReader((ROOT / "run/results.csv").open(encoding="utf-8")))
rows = [{k.strip(): v for k, v in r.items()} for r in rows]
fitness = lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])  # Ultralytics' selection rule
best = max(rows, key=fitness)
assert len(rows) == 27 and int(best["epoch"]) == 17, (len(rows), best["epoch"])

ch = json.loads((ROOT / "eval/candidate_640/post_v1_challenge.json").read_text(encoding="utf-8"))["rows"]
assert len(ch) == 4 and not any(r["yellowing_detected_at_0_25"] for r in ch), "post-V1 Yellowing challenge should be 0/4"
src = json.loads((ROOT / "SOURCE.json").read_text(encoding="utf-8"))
assert src["verdict"] == "CANDIDATE_MIXED" and src["source_experiment_commit"] == "2f2849c7de94df0b7a416c14eca6b9c40456332e"

if "--load" in sys.argv:
    from ultralytics import YOLO
    assert YOLO(str(ROOT / "weights/best.pt")).names == {0: "Healthy", 1: "Yellowing", 2: "Wilting"}

print(f"OK: {n} files match checksums; best.pt/last.pt hashes match; 27 epochs, best epoch 17; "
      "post-V1 Yellowing 0/4 at 640; verdict CANDIDATE_MIXED")

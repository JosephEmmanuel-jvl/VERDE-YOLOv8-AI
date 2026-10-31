"""Self-check for this package; run from anywhere: python verify_package.py (needs Pillow). Exits non-zero on failure."""
import hashlib, json, os
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for line in open(os.path.join(ROOT, "SHA256SUMS.txt")):
    digest, rel = line.rstrip("\n").split("  ", 1)
    assert sha(os.path.join(ROOT, rel)) == digest, f"checksum mismatch: {rel}"
records = json.load(open(os.path.join(ROOT, "manifests/annotations.json")))
assert len(records) == 4 and len({r["LEAKAGE_GROUP_ID"] for r in records}) == 4 and len({r["source_family"] for r in records}) == 2
assert sorted(os.listdir(os.path.join(ROOT, "images/yellowing"))) == sorted(os.path.basename(r["image"]) for r in records)
for r in records:
    im = Image.open(os.path.join(ROOT, r["image"])); im.load()
    W, H = ImageOps.exif_transpose(im).size
    assert [W, H] == r["image_size_wh"] and im.size == (W, H), f"orientation/size mismatch: {r['image']}"
    lines = open(os.path.join(ROOT, r["label"])).read().strip().splitlines()
    assert len(lines) == 1, r["label"]
    c, cx, cy, w, h = lines[0].split()
    cx, cy, w, h = map(float, (cx, cy, w, h))
    assert c == "1" and all(0 <= v <= 1 for v in (cx, cy, w, h)), r["label"]
    box = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
    assert all(abs(a - b) <= 1.5 for a, b in zip(box, r["bbox_xyxy"])), f"box mismatch: {r['label']}"
    assert r["class_status"] == "HUMAN_CLASS_APPROVED_YELLOWING" and r["box_status"] == "HUMAN_BOX_APPROVED" and r["role"] == "POST_CONDITION3_V1_TRAIN_CANDIDATE"
print("OK: 4 images, 4 boxes, 4 plant/group records, 2 source-family blocks; checksums and labels valid")

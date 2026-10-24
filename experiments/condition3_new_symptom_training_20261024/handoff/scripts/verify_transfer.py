"""Portable read-only verifier. Never installs software or calls model.train()."""
import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

BASELINE = '54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987'
GROUPS = {'LETTUCEDEF_GREENHOUSE_COHORT_20250410_20250605': (1, 6, 6),
          'LAB_SHARED_EXPERIMENT_20251125_20251210': (2, 6, 36)}
NEW_IDS = {'LETTUCEDEF_001', 'LETTUCEDEF_004', 'LETTUCEDEF_005', 'LETTUCEDEF_006',
           'LETTUCEDEF_010', 'LETTUCEDEF_016', 'LAB_Cam1', 'LAB_Cam5', 'LAB_Cam13',
           'LAB_Cam17', 'LAB_Cam21', 'LAB_Cam22'}
NAMES = {0: 'Healthy', 1: 'Yellowing', 2: 'Wilting'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(root, relative):
    return json.loads(safe(root, relative).read_text(encoding='utf-8'))


def safe(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and '\\' not in relative
            and ':' not in relative and str(name) == relative, f'Unsafe path: {relative}')
    path = root.joinpath(*name.parts)
    require(path.resolve().is_relative_to(root.resolve()), f'Path escapes package: {relative}')
    require(not path.is_symlink(), f'Symlink forbidden: {relative}')
    return path


def checksum_entries(root):
    entries = {}
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Bad checksum line')
        safe(root, name)
        require(name not in entries and name != 'SHA256SUMS.txt', f'Duplicate/self checksum: {name}')
        entries[name] = digest
    return entries


def check_zip(root, archive, expected, entries):
    require(len(expected) == 64 and sha(archive) == expected.lower(), 'TRANSFER ZIP SHA-256 MISMATCH')
    prefix = root.name + '/'
    with zipfile.ZipFile(archive) as zf:
        files = [i for i in zf.infolist() if not i.is_dir()]
        require(len({i.filename for i in files}) == len(files), 'Duplicate ZIP members')
        wanted = {prefix + p for p in entries} | {prefix + 'SHA256SUMS.txt'}
        require({i.filename for i in files} == wanted, 'ZIP inventory differs from package')
        for index, item in enumerate(files):
            relative = item.filename.removeprefix(prefix)
            safe(root, relative)
            digest = hashlib.sha256()
            # Reading to EOF also enforces each ZIP member's CRC.
            with zf.open(item) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            wanted_sha = sha(root / 'SHA256SUMS.txt') if relative == 'SHA256SUMS.txt' else entries[relative]
            require(digest.hexdigest() == wanted_sha, f'ZIP member SHA mismatch: {relative}')
            if index % 400 == 0:
                print(f'ZIP CRC/SHA {index + 1}/{len(files)}', flush=True)


def parse_yolo(path, width, height):
    boxes = []
    lines = path.read_text(encoding='utf-8').splitlines()
    require(bool(lines), f'Empty annotation: {path.name}')
    for line in lines:
        fields = line.split()
        require(len(fields) == 5 and fields[0] in ('0', '1', '2'), f'Malformed/class ID: {path}')
        x, y, w, h = map(float, fields[1:])
        require(all(math.isfinite(v) for v in (x, y, w, h)), f'Nonfinite box: {path}')
        require(0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1, f'Invalid box: {path}')
        # Historical YOLO uses six decimal places; permit only rounding at image edges.
        require(x - w / 2 >= -0.000002 and x + w / 2 <= 1.000002
                and y - h / 2 >= -0.000002 and y + h / 2 <= 1.000002, f'Box leaves image: {path}')
        boxes.append((int(fields[0]), [(x - w / 2) * width, (y - h / 2) * height,
                                     (x + w / 2) * width, (y + h / 2) * height]))
    return boxes


def verify(root, archive=None, expected=None, check_environment=True):
    root = Path(root).resolve()
    entries = checksum_entries(root)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p != root / 'SHA256SUMS.txt'}
    require(actual == set(entries), f'Package inventory mismatch: missing={set(entries)-actual}, extra={actual-set(entries)}')
    for index, (relative, digest) in enumerate(entries.items()):
        require(sha(safe(root, relative)) == digest, f'INTERNAL SHA-256 MISMATCH: {relative}')
        if index % 400 == 0:
            print(f'Internal SHA {index + 1}/{len(entries)}', flush=True)
    if archive is not None:
        check_zip(root, Path(archive), expected or '', entries)
    from PIL import Image
    version = read_json(root, 'package_version.json')
    require(version['package_id'] == root.name and version['training_runs'] == 0, 'Package identity/status mismatch')
    require(version['model_status'] == 'EXPERIMENTAL_CONDITION3_CANDIDATE', 'Model status changed')
    require(version['data_gate'] == 'DATA_GATE_NOT_READY'
            and version['acceptance_gate'] == 'INDEPENDENT_ACCEPTANCE_NOT_READY', 'Gate changed')
    require(read_json(root, 'configs/class_map.json') == {str(k): v for k, v in NAMES.items()}, 'Class map mismatch')
    historical = read_json(root, 'manifests/historical_images.json')
    require(len(historical) == 1167, 'Historical count mismatch')
    require(Counter(r['split'] for r in historical) == {'train': 760, 'val': 207, 'test': 200}, 'Original split mismatch')
    require(Counter(tuple(r['classes']) for r in historical) == {(0,): 233, (1,): 549, (2,): 385}, 'Historical class count mismatch')
    require(len({r['record_id'] for r in historical}) == 1167, 'Historical record ID collision')
    with (root / 'provenance/historical_split_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require([(r['stem'], r['split'], r['classes']) for r in csv_rows] ==
            [(r['record_id'], r['split'], ';'.join(map(str, r['classes']))) for r in historical], 'Original split CSV mismatch')
    source = read_json(root, 'provenance/source_snapshot.json')
    require(sha(root / 'provenance/historical_split_manifest.csv') == source['source_manifest_sha256'], 'Original manifest changed')
    hist_box_count = 0
    for index, record in enumerate(historical):
        image = safe(root, 'historical/' + record['image_relative_path'])
        label = safe(root, 'historical/' + record['label_relative_path'])
        require(sha(image) == record['image_sha256'] and sha(label) == record['label_sha256'], f'Historical file changed: {record["record_id"]}')
        with Image.open(image) as im:
            require(im.size == (record['width'], record['height']), 'Image dimensions mismatch')
            im.verify()
        boxes = parse_yolo(label, record['width'], record['height'])
        require(len(boxes) == record['boxes'] and sorted({b[0] for b in boxes}) == record['classes'], 'Historical annotation mismatch')
        hist_box_count += len(boxes)
        if index % 400 == 0:
            print(f'Image/annotation QA {index + 1}/1167', flush=True)
    require(hist_box_count == 1282, 'Historical box count mismatch')
    test_rows = [r for r in historical if r['split'] == 'test']
    require(read_json(root, 'manifests/historical_test_frozen.json') == test_rows, 'Frozen test manifest mismatch')
    test_hashes = {r['image_sha256'] for r in test_rows}
    require(not any(r['image_sha256'] in test_hashes for r in historical if r['split'] in ('train', 'val')), 'Historical development/test exact duplicate')
    evidence_root = root / 'new_symptom_evidence'
    ledger = read_json(evidence_root, 'metadata/provenance_annotation_ledger.json')
    require(len(ledger) == 12 and {r['image_id'] for r in ledger} == NEW_IDS, 'New-data allowlist mismatch')
    new_rows = read_json(root, 'manifests/new_symptom_images.json')
    require(len(new_rows) == 12 and {r['record_id'] for r in new_rows} == NEW_IDS, 'New source manifest mismatch')
    new_by_id = {r['record_id']: r for r in new_rows}
    groups = defaultdict(list)
    clipped = ambiguous = 0
    for record in ledger:
        rid = record['image_id']; new = new_by_id[rid]
        image = safe(evidence_root, record['package_original'])
        label = safe(evidence_root, record['yolo_annotation'])
        any_path = safe(evidence_root, record['anylabeling_annotation'])
        require(sha(image) == record['sha256'] == new['image_sha256'], f'New image changed: {rid}')
        require(sha(label) == new['label_sha256'] and sha(any_path) == new['anylabeling_sha256'], f'Approved annotations changed: {rid}')
        require(record['sha256'] not in test_hashes, f'NEW DATA EXACT TEST DUPLICATE: {rid}')
        require(record['human_class_approved'] and record['human_annotation_approved']
                and record['human_class_status'] == 'HUMAN_CLASS_APPROVED'
                and record['annotation_status'] == 'HUMAN_BOX_APPROVED', f'Human approval missing: {rid}')
        group = record['LEAKAGE_GROUP_ID']
        require(group in GROUPS and record['class_id'] == GROUPS[group][0], 'Group/class mismatch')
        require(new['leakage_group'] == group and new['role'] == 'EXPERIMENTAL_TRAIN_CANDIDATE'
                and new['proposed_split'] == 'train' and not new['independent_acceptance_evidence'], 'New role mismatch')
        with Image.open(image) as im:
            require(im.size == (record['image_width'], record['image_height']), 'New dimensions mismatch')
            im.verify()
        boxes = parse_yolo(label, record['image_width'], record['image_height'])
        any_data = json.loads(any_path.read_text(encoding='utf-8'))
        require((any_path.parent / any_data['imagePath']).resolve() == image.resolve(), 'AnyLabeling image path mismatch')
        require(any_data['imageWidth'] == record['image_width'] and any_data['imageHeight'] == record['image_height']
                and any_data['source_sha256'] == record['sha256']
                and any_data['LEAKAGE_GROUP_ID'] == group
                and any_data['annotation_status'] == 'HUMAN_BOX_APPROVED', 'AnyLabeling metadata mismatch')
        require(any_data['intentionally_unboxed_ambiguity_metadata'] == record['ambiguous_plant_observations'], 'Ambiguity metadata changed')
        require(len(boxes) == len(record['boxes']) == len(any_data['shapes']) == record['number_of_boxes'], 'Box count mismatch')
        for (cid, xyxy), approved, shape in zip(boxes, record['boxes'], any_data['shapes']):
            require(cid == approved['class_id'] == record['class_id'] and approved['human_annotation_approved']
                    and approved['annotation_status'] == 'HUMAN_BOX_APPROVED', 'Box approval/class mismatch')
            require(all(abs(a-b) <= .03 for a, b in zip(xyxy, approved['bbox_xyxy'])), 'YOLO/approved geometry mismatch')
            require(shape['shape_type'] == 'rectangle' and shape['label'] == NAMES[cid]
                    and [v for point in shape['points'] for v in point] == approved['bbox_xyxy'], 'AnyLabeling geometry mismatch')
            require(shape['flags'].get('HUMAN_BOX_APPROVED') is True and
                    bool(shape['flags'].get('FRAME_CLIPPED_PLANT', False)) == approved['frame_clipped'], 'Box flags changed')
            clipped += int(approved['frame_clipped'])
        ambiguous += len(record['ambiguous_plant_observations'])
        require(all(a['boxed'] is False and a['bbox_xyxy'] is None for a in record['ambiguous_plant_observations']), 'Ambiguity was silently boxed')
        groups[group].append(record)
    require(clipped == 20 and ambiguous == 3, 'Clipping/ambiguity count changed')
    registry = read_json(root, 'manifests/leakage_groups.json')
    require(len(registry) == 2 and {g['leakage_group'] for g in registry} == set(GROUPS), 'Group registry mismatch')
    for group, (cid, images, boxes) in GROUPS.items():
        require(len(groups[group]) == images and sum(r['number_of_boxes'] for r in groups[group]) == boxes, 'Group cardinality mismatch')
        matching = next(r for r in registry if r['leakage_group'] == group)
        require(matching['record_ids'] == sorted(r['image_id'] for r in groups[group])
                and matching['role'] == 'EXPERIMENTAL_TRAIN_CANDIDATE' and matching['split'] == 'train'
                and matching['independent_acceptance_groups'] == 0, 'Group role/membership changed')
    plan = read_json(root, 'manifests/proposed_experiment_build.json')
    expected_rows = [dict(record_id=r['record_id'], origin='historical', split=r['split'],
                          image='historical/' + r['image_relative_path'], label='historical/' + r['label_relative_path'],
                          image_sha256=r['image_sha256'], label_sha256=r['label_sha256'], leakage_group=None,
                          role='HISTORICAL_' + r['split'].upper()) for r in historical]
    expected_rows += [dict(record_id=r['record_id'], origin='new_symptom', split='train', image=r['image'], label=r['label'],
                          image_sha256=r['image_sha256'], label_sha256=r['label_sha256'], leakage_group=r['leakage_group'],
                          role='EXPERIMENTAL_TRAIN_CANDIDATE') for r in new_rows]
    require(plan['records'] == expected_rows, 'Proposed allocation/source mapping changed')
    require(plan['counts'] == {'train': 772, 'val': 207, 'test': 200, 'total': 1179}, 'Proposed counts mismatch')
    for group in GROUPS:
        allocated = [r for r in plan['records'] if r['leakage_group'] == group]
        require({r['split'] for r in allocated} == {'train'} and
                {r['role'] for r in allocated} == {'EXPERIMENTAL_TRAIN_CANDIDATE'}, 'NEW GROUP CROSSES ROLES')
    require(sha(root / 'models/verde_condition3_best_validation.pt') == BASELINE, 'BASELINE CHECKPOINT MISMATCH')
    require(read_json(root, 'models/baseline_checkpoint.json')['sha256'] == BASELINE, 'Baseline reference mismatch')
    result = dict(result='PASS', internal_checksum_files=len(entries), historical_images=1167, historical_boxes=1282,
                  new_records=12, new_boxes=42, new_groups=2, frame_clipped_boxes=20, unboxed_ambiguities=3,
                  independent_acceptance_groups=0, test_images_preserved=200, training_runs=0,
                  zip_sha_crc='PASS' if archive else 'NOT_REQUESTED_INTERNAL_ONLY')
    if check_environment:
        import torch
        import ultralytics
        require(torch.cuda.is_available(), 'CUDA UNAVAILABLE: use a CUDA-enabled PyTorch environment')
        device = torch.cuda.get_device_properties(0)
        require(torch.version.cuda is not None, 'NVIDIA CUDA runtime not detected')
        require(torch.cuda.device_count() >= 1, 'No CUDA device')
        a = torch.ones((16, 16), device='cuda')
        require(float((a @ a)[0, 0].item()) == 16, 'CUDA smoke test failed')
        model = ultralytics.YOLO(str(root / 'models/verde_condition3_best_validation.pt'))
        require({int(k): v for k, v in model.names.items()} == NAMES, 'Checkpoint class map mismatch')
        result['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     ultralytics=ultralytics.__version__, gpu=device.name,
                                     vram_bytes=device.total_memory, checkpoint_loading='PASS', cuda_smoke='PASS')
    else:
        result['environment'] = 'SKIPPED_PACKAGE_ONLY_MODE_NOT_A_GPU_PASS'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--expected-zip-sha256')
    parser.add_argument('--package-only', action='store_true', help='Skip GPU check; never counts as a GPU pass')
    parser.add_argument('--internal-only', action='store_true', help='Publisher staging only; does not validate ZIP')
    args = parser.parse_args()
    require(args.internal_only or (args.zip is not None and args.expected_zip_sha256 is not None),
            'Provide --zip and --expected-zip-sha256 to verify the transferred ZIP')
    result = verify(Path(__file__).resolve().parents[1], args.zip, args.expected_zip_sha256, not args.package_only)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    try:
        main()
    except Exception as exc:
        print(f'VERDE VERIFICATION FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)

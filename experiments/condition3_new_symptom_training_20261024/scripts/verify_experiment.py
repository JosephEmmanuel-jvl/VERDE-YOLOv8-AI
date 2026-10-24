"""Verify this additive repository experiment; never train or modify source data."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
BASELINE = '54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987'
GROUPS = {1: 'LETTUCEDEF_GREENHOUSE_COHORT_20250410_20250605',
          2: 'LAB_SHARED_EXPERIMENT_20251125_20251210'}
IDS = {'LETTUCEDEF_001', 'LETTUCEDEF_004', 'LETTUCEDEF_005', 'LETTUCEDEF_006',
       'LETTUCEDEF_010', 'LETTUCEDEF_016', 'LAB_Cam1', 'LAB_Cam5', 'LAB_Cam13',
       'LAB_Cam17', 'LAB_Cam21', 'LAB_Cam22'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and ':' not in name
            and '\\' not in name and str(relative) == name, 'Unsafe manifest path: ' + name)
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Unsafe resolved path')
    return path


def entries(root, filename='SHA256SUMS.txt'):
    result = {}
    for line in (root / filename).read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        require(name not in result and len(digest) == 64, 'Malformed checksum manifest')
        safe(root, name); result[name] = digest
    return result


def load(root, name):
    return json.loads(safe(root, name).read_text(encoding='utf-8'))


def verify(root, repo=None, historical=None):
    root = root.resolve(); repo = (repo or root.parents[1]).resolve()
    manifest = entries(root)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'SHA256SUMS.txt'}
    require(actual == set(manifest), 'Experiment exact-inventory mismatch')
    for name, digest in manifest.items():
        require(sha(safe(root, name)) == digest, 'EXPERIMENT SHA MISMATCH: ' + name)
    full = root / 'handoff'
    info = load(root, 'metadata/publication.json')
    require(info['representation'] == 'SPARSE_REPOSITORY_DOCUMENTATION_NOT_FULL_TRANSFER_WORKSPACE', 'Representation changed')
    require(info['status'] == ['EXPERIMENTAL', 'NOT CANONICAL', 'NOT DEPLOYMENT READY', 'DO_NOT_DEPLOY'], 'Status changed')
    require(info['data_gate'] == 'DATA_GATE_NOT_READY' and info['acceptance_gate'] == 'INDEPENDENT_ACCEPTANCE_NOT_READY', 'Gates changed')
    require(info['zip_uploaded'] is False and info['training_runs'] == 0, 'Publication scope changed')
    reference = entries(full)
    require(sha(full/'SHA256SUMS.txt') == info['handoff_manifest_sha256'], 'Original handoff manifest changed')
    omitted = []
    for name, digest in reference.items():
        if name.startswith('historical/') or name == 'models/verde_condition3_best_validation.pt':
            require(not (full/name).exists(), 'Historical/model binary duplicated in experiment')
            omitted.append(name)
        else:
            require(sha(safe(full, name)) == digest, 'Authoritative handoff copy changed: ' + name)
    require(len(reference) == 2404 and len(omitted) == 2335, 'Handoff payload coverage changed')
    require({p.relative_to(full).as_posix() for p in full.rglob('*') if p.is_file()} ==
            (set(reference)-set(omitted)) | {'SHA256SUMS.txt'}, 'Sparse handoff inventory mismatch')
    require(sha(repo/'models/verde_condition3_best_validation.pt') == BASELINE, 'REPOSITORY BASELINE HASH MISMATCH')
    require(load(full, 'configs/class_map.json') == {'0':'Healthy','1':'Yellowing','2':'Wilting'}, 'Class map mismatch')
    source = load(full, 'manifests/new_symptom_images.json')
    ledger = load(full, 'new_symptom_evidence/metadata/provenance_annotation_ledger.json')
    require(len(ledger) == len(source) == 12 and {r['image_id'] for r in ledger} == IDS, 'Unexpected new evidence')
    spec = importlib.util.spec_from_file_location('transfer_geometry', full/'scripts/verify_transfer.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    from PIL import Image
    counts = Counter(); clipped = ambiguous = 0
    for record in ledger:
        rid = record['image_id']; cid = record['class_id']
        require(cid in GROUPS and record['LEAKAGE_GROUP_ID'] == GROUPS[cid], 'Leakage/class mismatch')
        require(record['human_class_status'] == 'HUMAN_CLASS_APPROVED' and record['human_class_approved']
                and record['human_annotation_approved'] and record['annotation_status'] == 'HUMAN_BOX_APPROVED', 'Human approval missing')
        require(record['role'] == 'TRAIN_CANDIDATE' and not record['independent_acceptance_evidence']
                and not record['training_authorized'], 'Acceptance/training scope changed')
        require(record['license'] == 'CC BY 4.0' and record['license_url'] == 'https://creativecommons.org/licenses/by/4.0/', 'Rights metadata mismatch')
        evidence = full / 'new_symptom_evidence'
        image = safe(evidence, record['package_original'])
        require(sha(image) == record['sha256'], 'Image SHA mismatch')
        with Image.open(image) as im:
            require(im.size == (record['image_width'], record['image_height']), 'Dimensions mismatch'); im.verify()
        boxes = helper.parse_yolo(safe(evidence,record['yolo_annotation']), record['image_width'],record['image_height'])
        require(sha(safe(evidence,record['yolo_annotation'])) == record['source_yolo_sha256'], 'Approved YOLO bytes changed')
        any_path = safe(evidence,record['anylabeling_annotation']); any_data = json.loads(any_path.read_text(encoding='utf-8'))
        require((any_path.parent/any_data['imagePath']).resolve() == image.resolve(), 'AnyLabeling image linkage changed')
        require(any_data['annotation_status'] == 'HUMAN_BOX_APPROVED'
                and any_data['intentionally_unboxed_ambiguity_metadata'] == record['ambiguous_plant_observations'], 'Approval/ambiguity changed')
        require(len(boxes) == len(record['boxes']) == len(any_data['shapes']) == record['number_of_boxes'], 'Box count mismatch')
        for (box_class, xyxy), approved, shape in zip(boxes,record['boxes'],any_data['shapes']):
            require(box_class == approved['class_id'] == cid and approved['annotation_status'] == 'HUMAN_BOX_APPROVED', 'Box class/approval mismatch')
            require(max(abs(a-b) for a,b in zip(xyxy,approved['bbox_xyxy'])) < .03, 'YOLO geometry changed')
            require([v for point in shape['points'] for v in point] == approved['bbox_xyxy']
                    and shape['label'] == record['class_name'] and shape['shape_type'] == 'rectangle', 'AnyLabeling geometry changed')
            clipped += int(approved['frame_clipped'])
        ambiguous += len(record['ambiguous_plant_observations'])
        counts[(cid,'images')] += 1; counts[(cid,'boxes')] += len(boxes)
    require(counts == {(1,'images'):6,(1,'boxes'):6,(2,'images'):6,(2,'boxes'):36} and clipped == 20 and ambiguous == 3, 'Evidence cardinality changed')
    historical_rows = load(full,'manifests/historical_images.json')
    require(len(historical_rows) == 1167 and Counter(r['split'] for r in historical_rows) == {'train':760,'val':207,'test':200}, 'Historical definition changed')
    require(Counter(tuple(r['classes']) for r in historical_rows) == {(0,):233,(1,):549,(2,):385}, 'Historical class count changed')
    test = [r for r in historical_rows if r['split']=='test']
    require(load(full,'manifests/historical_test_frozen.json') == test, 'Frozen test mapping changed')
    test_hashes = {r['image_sha256'] for r in test}
    plan = load(full,'manifests/proposed_experiment_build.json')
    require(plan['counts'] == {'train':772,'val':207,'test':200,'total':1179} and len(plan['records']) == 1179, 'Proposed allocation changed')
    require(not any(r['image_sha256'] in test_hashes for r in plan['records'] if r['split']=='train'), 'EXACT TEST DUPLICATE IN TRAINING')
    require(Counter(r['split'] for r in plan['records']) == {'train':772,'val':207,'test':200}, 'Allocation cardinality mismatch')
    for group in GROUPS.values():
        grouped = [r for r in plan['records'] if r['leakage_group']==group]
        require(len(grouped)==6 and {r['split'] for r in grouped}=={'train'} and
                {r['role'] for r in grouped}=={'EXPERIMENTAL_TRAIN_CANDIDATE'}, 'NEW GROUP CROSSES ROLES')
    if historical is not None:
        for record in historical_rows:
            for key, digest in [('image_relative_path','image_sha256'),('label_relative_path','label_sha256')]:
                require(sha(safe(historical,record[key])) == record[digest], 'Restored historical file mismatch: '+record[key])
    rights = load(root,'rights/verification.json')
    require(rights['redistribution'] == 'PERMITTED_CC_BY_4_0_WITH_ATTRIBUTION' and rights['verified_records']==12, 'Redistribution record missing')
    return dict(result='PASS', committed_checksum_files=len(manifest), authoritative_payload_entries=2404,
                historical_payload_files_intentionally_reused=2334, baseline_reused=True,
                yellowing_images=6,yellowing_boxes=6,wilting_scenes=6,wilting_boxes=36,new_groups=2,
                frame_clipped_boxes=20,intentionally_unboxed_ambiguities=3,independent_acceptance_groups=0,
                historical_test_definition='200 PAIRS HASH-FROZEN; NOT PROVEN INDEPENDENT',
                historical_local_files='PASS' if historical is not None else 'NOT_REQUESTED_METADATA_ONLY',
                baseline_sha256=BASELINE, training_runs=0, gpu_verification='NOT_RUN_BY_METADATA_VERIFIER; use full transfer verifier on receiving laptop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--historical',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1],args.repo,args.historical),indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'VERDE EXPERIMENT VERIFICATION FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)

"""Construct NEW isolated dataset copies from a verified package. Never trains."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
from verify_transfer import require, sha, verify


def build(root, destination, variant, acknowledge_partial=False, plan_only=False):
    root = root.resolve(); destination = destination.resolve()
    require(not destination.exists(), 'Destination must be NEW; existing directories are never overwritten')
    require(not destination.is_relative_to(root) and not root.is_relative_to(destination), 'Destination must be outside package')
    require(destination.parent.exists(), 'Destination parent must already exist')
    plan = json.loads((root / 'manifests/proposed_experiment_build.json').read_text(encoding='utf-8'))
    rows = [r for r in plan['records'] if variant == 'plus-new' or r['origin'] == 'historical']
    if variant == 'plus-new' and not plan_only:
        require(acknowledge_partial, 'LAB has 3 intentionally unboxed ambiguous edge regions. Explicit --acknowledge-partial-scene-risk is required for the proposed full-scene experiment; this does not create an ignore mask or certify fully labeled scenes.')
    names = [f'{r["split"]}/{r["record_id"]}' for r in rows]
    require(len(names) == len(set(names)), 'Output filename collision')
    summary = dict(variant=variant, images=len(rows), train=sum(r['split']=='train' for r in rows),
                   val=sum(r['split']=='val' for r in rows), test=sum(r['split']=='test' for r in rows),
                   new_records=sum(r['origin']=='new_symptom' for r in rows), training_runs=0,
                   partial_scene_risk_acknowledged=acknowledge_partial)
    if plan_only:
        return summary
    destination.mkdir()
    for index, row in enumerate(rows):
        for source_key, category in [('image', 'images'), ('label', 'labels')]:
            source = root / row[source_key]
            target = destination / category / row['split'] / (row['record_id'] + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(sha(target) == row[source_key + '_sha256'], f'Constructed file hash mismatch: {target}')
        if index % 400 == 0:
            print(f'Copied isolated dataset {index + 1}/{len(rows)}', flush=True)
    (destination / 'data.yaml').write_text('path: ' + json.dumps(destination.as_posix()) +
        '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Healthy\n  1: Yellowing\n  2: Wilting\n', encoding='utf-8')
    receipt = dict(summary, created_at=datetime.now(timezone.utc).isoformat(),
                   source_package=root.name, source_checksum_manifest_sha256=sha(root / 'SHA256SUMS.txt'),
                   source_build_plan_sha256=sha(root / 'manifests/proposed_experiment_build.json'),
                   records=rows, model_status='EXPERIMENTAL_CONDITION3_CANDIDATE',
                   data_gate='DATA_GATE_NOT_READY', acceptance_gate='INDEPENDENT_ACCEPTANCE_NOT_READY',
                   warning='LAB unboxed fragments may be treated as background by stock YOLO; acknowledgment records experimental risk only. No annotation or ignore mask was invented.')
    (destination / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    shutil.copyfile(root / 'configs/training_template.yaml', destination / 'training_template.yaml')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--variant', choices=['historical-control', 'plus-new'], default='plus-new')
    parser.add_argument('--acknowledge-partial-scene-risk', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--expected-zip-sha256', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify(root, args.zip, args.expected_zip_sha256, check_environment=True)
    print(json.dumps(build(root, args.destination, args.variant, args.acknowledge_partial_scene_risk, args.plan_only), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'VERDE BUILD REFUSED/FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)

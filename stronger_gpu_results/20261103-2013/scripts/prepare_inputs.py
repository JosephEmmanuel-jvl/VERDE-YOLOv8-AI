"""Verify the immutable restore, then create disposable byte-verified working copies."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import torch
import yaml

OUT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/VERDE-restore/field-adaptation-savepoint-20261103-190647')
WORK = OUT / '_work'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    assert not WORK.exists(), 'Refusing to overwrite working inputs'
    manifest = SOURCE / 'FULL_SHA256SUMS.txt'
    verified = {}
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = SOURCE / relative.lstrip('*').replace('\\', '/')
        assert SOURCE.resolve() in path.resolve().parents, relative
        assert sha(path) == expected.lower(), relative
        verified[path.relative_to(SOURCE).as_posix()] = expected.lower()
    snapshot = {p.relative_to(SOURCE).as_posix(): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    (OUT / 'source_integrity_before.json').write_text(json.dumps({'source': str(SOURCE), 'full_manifest_entries_verified': len(verified), 'all_files': snapshot}, indent=2)+'\n')
    print('Verified FULL_SHA256SUMS:', len(verified), 'and snapshotted', len(snapshot), 'source files', flush=True)
    WORK.mkdir()
    copies = {}
    for relative, target in [('morning-yellowing-training/candidate_dataset', 'morning_dataset'), ('field-adaptation/dataset-v2', 'field_dataset')]:
        src, dst = SOURCE / relative, WORK / target
        shutil.copytree(src, dst)
        files = [p for p in src.rglob('*') if p.is_file()]
        for path in files:
            assert sha(path) == sha(dst/path.relative_to(src)), str(path)
        copies[target] = {'source': relative, 'verified_files': len(files)}
    configs = OUT / 'configs'
    configs.mkdir()
    for name, dataset in [('morning', WORK/'morning_dataset'), ('condition', WORK/'field_dataset/condition_detect'), ('prop', WORK/'field_dataset/prop_detect')]:
        data = yaml.safe_load((dataset/'data.yaml').read_text())
        data['path'] = dataset.as_posix()
        (configs / f'{name}.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    # The working datasets retain exact bytes, including original YAML. Only separate result configs relocate paths.
    pristine = OUT / 'preserved_inputs'
    pristine.mkdir()
    for name, expected in [('best', '83f2f0e9522a8c5e7258fcab7ccf4019205c8d549a956657d814b63adb1ff730'), ('last', '147c39ddc83a72e9b1d5e45e0cb6eab334ea07e5ad154fc08d3250f7c671f55d')]:
        src = SOURCE/f'morning-yellowing-training/weights/{name}.pt'
        assert sha(src) == expected
        shutil.copy2(src, pristine/f'morning_original_{name}.pt')
    run = OUT / 'task_a'
    (run/'weights').mkdir(parents=True)
    shutil.copy2(pristine/'morning_original_best.pt', run/'weights/best.pt')
    shutil.copy2(pristine/'morning_original_last.pt', run/'weights/last.pt')
    shutil.copy2(SOURCE/'morning-yellowing-training/run/results.csv', run/'results.csv')
    shutil.copy2(SOURCE/'morning-yellowing-training/run/args.yaml', pristine/'morning_original_args.yaml')
    shutil.copy2(SOURCE/'field-adaptation/baseline-v2/baseline-report.json', pristine/'baseline-v2.json')
    shutil.copy2(SOURCE/'field-adaptation/dataset-v2/DATASET_SUMMARY.json', pristine/'DATASET_SUMMARY.json')
    checkpoint = torch.load(pristine/'morning_original_last.pt', map_location='cpu', weights_only=False)
    rows = list(csv.DictReader((run/'results.csv').open()))
    assert len(rows) == 11 and int(rows[-1]['epoch']) == 11
    assert checkpoint['epoch'] == 10 and checkpoint.get('optimizer') is not None
    assert checkpoint['train_args']['epochs'] == 20
    details = {'copies': copies, 'checkpoint_epoch_zero_based': checkpoint['epoch'], 'start_epoch_one_based': checkpoint['epoch']+2, 'optimizer_state_present': checkpoint.get('optimizer') is not None, 'ema_present': checkpoint.get('ema') is not None, 'checkpoint_version': checkpoint.get('version'), 'train_args': checkpoint['train_args'], 'completed_rows': len(rows), 'field': {}}
    for task in ['condition_detect', 'prop_detect', 'harvest_classify']:
        details['field'][task] = {}
        for split, expected in [('train', 180 if task != 'prop_detect' else 30), ('val', 60 if task != 'prop_detect' else 10), ('test', 240 if task != 'prop_detect' else 40)]:
            folder = WORK/'field_dataset'/task/(f'images/{split}' if task != 'harvest_classify' else split)
            images = sorted(folder.rglob('*.jpg'))
            assert len(images) == expected, (task, split, len(images))
            for image in images:
                if split in ['train', 'val']:
                    assert image.name.startswith(('set-01-', 'set-03-')), image
                else:
                    assert image.name.startswith(('set-02-', 'set-04-')), image
            details['field'][task][split] = len(images)
    (OUT/'input_validation.json').write_text(json.dumps(details, indent=2, default=str)+'\n')
    print(json.dumps({k:v for k,v in details.items() if k != 'train_args'}, indent=2), flush=True)

if __name__ == '__main__':
    main()

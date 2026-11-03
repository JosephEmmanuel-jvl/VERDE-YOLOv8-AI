"""Check preserved frame boundaries and disclose exact cross-split duplicates without reassignment."""
import json
import re
from collections import defaultdict
from prepare_inputs import OUT, WORK

snapshot = json.loads((OUT/'source_integrity_before.json').read_text())['all_files']
report = {'frame_boundary_violations': [], 'duplicate_groups': {}}
for task in ['condition_detect', 'harvest_classify', 'prop_detect']:
    by_hash = defaultdict(list)
    for relative, digest in snapshot.items():
        prefix = f'field-adaptation/dataset-v2/{task}/'
        if not relative.startswith(prefix) or not relative.endswith('.jpg'):
            continue
        parts = relative[len(prefix):].split('/')
        split = parts[1] if task != 'harvest_classify' else parts[0]
        name = parts[-1]
        number = int(re.search(r'_(\d{2})_frame-', name).group(1))
        if split == 'train':
            good = name.startswith(('set-01-', 'set-03-')) and 1 <= number <= 15
        elif split == 'val':
            good = name.startswith(('set-01-', 'set-03-')) and 16 <= number <= 20
        else:
            good = split == 'test' and name.startswith(('set-02-', 'set-04-')) and 1 <= number <= 20
        if not good:
            report['frame_boundary_violations'].append(relative)
        by_hash[digest].append({'path':relative,'split':split})
    report['duplicate_groups'][task] = {h:rows for h,rows in by_hash.items() if len(rows)>1}
assert not report['frame_boundary_violations'], report
report['policy'] = 'No duplicate removed, relabelled, or reassigned. Any test overlap is disclosed; no independent acceptance claim.'
(OUT/'split_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'frame_boundary_violations':0,'duplicate_group_counts':{k:len(v) for k,v in report['duplicate_groups'].items()}, 'split_pairs':{k:[sorted(set(x['split'] for x in rows)) for rows in v.values()] for k,v in report['duplicate_groups'].items()}},indent=2))

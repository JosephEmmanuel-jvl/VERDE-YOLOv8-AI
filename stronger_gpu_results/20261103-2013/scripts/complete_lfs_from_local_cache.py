"""Reuse hash-verified objects from the existing laptop clone; no remote changes."""
import json
import shutil
import subprocess
from pathlib import Path
from prepare_inputs import OUT, sha

REPO = OUT.parents[1]
OLD = Path('C:/Users/ben/Documents/Codex/VERDE-YOLOv8-AI/.git/lfs/objects')
rows = []
for line in subprocess.check_output(['git','lfs','ls-files','--long'], cwd=REPO, text=True).splitlines():
    oid, marker, relative = line.split(maxsplit=2)
    target = REPO/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
    if target.is_file():
        assert sha(target)==oid
        rows.append({'path':relative,'oid':oid,'status':'VERIFIED_PRESENT'})
        continue
    source = OLD/oid[:2]/oid[2:4]/oid
    if not source.is_file():
        rows.append({'path':relative,'oid':oid,'status':'NOT_IN_EXISTING_LOCAL_CACHE'})
        continue
    assert sha(source)==oid
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert sha(target)==oid
    rows.append({'path':relative,'oid':oid,'status':'VERIFIED_COPIED_FROM_EXISTING_LAPTOP_CACHE'})
    print('Verified cached object:',relative,flush=True)
(OUT/'lfs_local_cache_verification.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)

"""Reconstruct the exact handoff payload from repository metadata and restored data; never train."""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from verify_experiment import entries, require, safe, sha, verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination-parent',type=Path,required=True,help='Existing short local path; creates a NEW versioned child')
    parser.add_argument('--historical',type=Path,help='Existing prepared historical three_class directory; default repository dataset/three_class')
    parser.add_argument('--package-only',action='store_true',help='Skip receiving GPU test; never counts as GPU verification')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; repo=root.parents[1]
    historical=(args.historical or repo/'dataset/three_class').resolve()
    verify(root,repo,historical)
    sparse=root/'handoff'; info=json.loads((root/'metadata/publication.json').read_text(encoding='utf-8'))
    destination=args.destination_parent.resolve()/info['handoff_package_id']
    require(args.destination_parent.is_dir() and not destination.exists(),'Destination parent must exist and package destination must be NEW')
    require(not destination.is_relative_to(repo) and not repo.is_relative_to(destination),'Reconstruction must be outside repository')
    mapping=[]
    for name,digest in entries(sparse).items():
        if name.startswith('historical/'):
            source=safe(historical,name.removeprefix('historical/'))
        elif name=='models/verde_condition3_best_validation.pt':
            source=repo/name
        else:
            source=safe(sparse,name)
        require(source.is_file() and sha(source)==digest,'Missing/mismatched reconstruction input: '+name)
        require(os.name!='nt' or len(str(destination/name))<260,'Use a shorter destination parent for Windows paths')
        mapping.append((source,name,digest))
    require(len(mapping)==2404,'Unexpected full payload count')
    if args.plan_only:
        print(json.dumps({'result':'PASS_PLAN_ONLY','payload_files':2404,'destination':str(destination),'training_runs':0}));return
    destination.mkdir()
    for index,(source,name,digest) in enumerate(mapping):
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);require(sha(target)==digest,'Reconstruction copy mismatch: '+name)
        if index%400==0: print(f'Reconstructed {index+1}/2404',flush=True)
    shutil.copyfile(sparse/'SHA256SUMS.txt',destination/'SHA256SUMS.txt')
    spec=importlib.util.spec_from_file_location('full_transfer_check',destination/'scripts/verify_transfer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify(destination,check_environment=not args.package_only)
    result['reconstructed_payload']='BYTE_IDENTICAL_TO_ORIGINAL_2404_PAYLOAD_CHECKSUMS'
    result['original_zip']='NOT_RECREATED; original offline ZIP retains its own recorded SHA-256'
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'VERDE RECONSTRUCTION REFUSED/FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)

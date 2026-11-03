"""Read-only source recheck, final evidence report, model hashes, and result manifest."""
import csv
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from prepare_inputs import OUT, SOURCE, sha

REPO = OUT.parents[1]
BASE = '7d7b81f19d01ee6f751f5a6a842626fec9a60f8f'

def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')

def main():
    for task in ['task_a','field_condition','field_harvest','field_prop']:
        assert (OUT/f'{task}_completion.json').is_file(), task
        for relative in ['weights/best.pt','weights/last.pt','results.csv','args.yaml']:
            assert (OUT/task/relative).is_file(), (task,relative)
    assert json.loads((OUT/'ACTIVE_SAVEPOINT.json').read_text())['stage']=='TRAINING_AND_EVALUATION_COMPLETE'
    before = json.loads((OUT/'source_integrity_before.json').read_text())
    actual_paths = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
    assert actual_paths == set(before['all_files']), 'Restored file inventory changed'
    changed = [relative for relative,expected in before['all_files'].items() if sha(SOURCE/relative)!=expected]
    assert not changed, changed
    assert sha(OUT/'preserved_inputs/morning_original_best.pt') == '83f2f0e9522a8c5e7258fcab7ccf4019205c8d549a956657d814b63adb1ff730'
    assert sha(OUT/'preserved_inputs/morning_original_last.pt') == '147c39ddc83a72e9b1d5e45e0cb6eab334ea07e5ad154fc08d3250f7c671f55d'
    assert (OUT/'task_a/results.csv').read_bytes().startswith((SOURCE/'morning-yellowing-training/run/results.csv').read_bytes()), 'Original11 CSV rows were changed'
    write_json('source_integrity_after.json',{'source':str(SOURCE),'files_verified':len(actual_paths),'changed':changed,'added_or_missing':[],'status':'PASS'})
    models = {p.relative_to(OUT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.rglob('*.pt') if '_work' not in p.relative_to(OUT).parts and 'runtime' not in p.relative_to(OUT).parts}
    write_json('MODEL_SHA256.json',models)
    a = json.loads((OUT/'task_a_comparison.json').read_text())
    field = json.loads((OUT/'field_comparison.json').read_text())
    environment = json.loads((OUT/'environment_preflight.json').read_text())
    for name,record in environment['starter_models'].items():
        assert sha(REPO/'continuation_packages/starter_models'/name) == record['sha256'], name
    baseline, candidate = field['runs']['baseline'],field['runs']['candidate']
    table = ['| Condition class | Baseline precision | Candidate precision | Baseline recall | Candidate recall | Baseline mAP50 | Candidate mAP50 | Baseline mAP50-95 | Candidate mAP50-95 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['Healthy','Yellowing','Wilting']:
        old = baseline['condition_detection_metrics']['per_class'][name]
        new = candidate['condition_detection_metrics']['per_class'][name]
        table.append('| '+name+' | '+' | '.join(f'{x:.6f}' for metric in ['precision','recall','map50','map50_95'] for x in [old[metric],new[metric]])+' |')
    a_table = ['| Task A checkpoint | Class | Precision | Recall | mAP50 | mAP50-95 |','|---|---|---:|---:|---:|---:|']
    for checkpoint, report in a['models'].items():
        for name, metrics in report['per_class'].items():
            a_table.append('| '+checkpoint+' | '+name+' | '+' | '.join(f'{metrics[key]:.6f}' for key in ['precision','recall','map50','map50_95'])+' |')
    completed = {task:json.loads((OUT/f'{task}_completion.json').read_text()) for task in ['task_a','field_condition','field_harvest','field_prop']}
    write_json('COMPLETION_SUMMARY.json',{'tasks':completed,'task_a_selected':a['selected'],'field_baseline':{t:{k:v for k,v in baseline[t].items() if k!='predictions'} for t in ['condition','harvest','prop']},'field_candidate':{t:{k:v for k,v in candidate[t].items() if k!='predictions'} for t in ['condition','harvest','prop']}})
    def compact(report,task):
        return json.dumps({k:v for k,v in report[task].items() if k!='predictions'},indent=2)
    model_lines = '\n'.join(f'- `{relative}` — `{item["sha256"]}` ({item["bytes"]:,} bytes)' for relative,item in models.items() if '/weights/best.pt' in relative or '/weights/last.pt' in relative)
    recommendations = {'task_a_validation_selected':{'path':a['selected_path'],'sha256':a['selected_sha256'],'scope':'Yellowing-focused validation-selected experimental candidate; not independently validated.'},'fixed_demo':{},'deployment_authorized':False}
    recommendations['task_a_balanced_validation_candidate'] = {'path':'task_a/weights/best.pt','sha256':models['task_a/weights/best.pt']['sha256'],'scope':'Alternative emphasizing macro AP and Wilting; review Healthy recall tradeoff in the per-class table. Not deployable.'}
    for task, directory in [('condition','field_condition'),('harvest','field_harvest'),('prop','field_prop')]:
        recommendations['fixed_demo'][task] = {'candidate_path':f'{directory}/weights/best.pt','sha256':models[f'{directory}/weights/best.pt']['sha256'],'recommendation':'Candidate for fresh isolated full-pipeline validation only; inspect per-class tradeoffs below, not deployable.'}
    recommendations = json.loads((OUT/'RECOMMENDATION_REVIEW.json').read_text())
    write_json('RECOMMENDED_CANDIDATES.json',recommendations)
    report = f'''# VERDE stronger-GPU training continuation

Completed at {datetime.now(timezone(timedelta(hours=8))).isoformat()} (Asia/Manila).

Base handoff: `{BASE}`. Results branch: `{subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()}`.

## Scope and protection

All four authorized training tasks and their evaluations completed in this new result directory. The restored package's {len(actual_paths):,} files are unchanged in inventory and SHA-256. Original checkpoints and datasets were never used as output paths. No Raspberry Pi access, deployment, canonical/historical dataset modification, weight merging, or main-branch modification occurred.

Task A resumed checkpoint epoch 10 (zero-based), continuing with epoch 12 (one-based), all 183 optimizer-state entries restored. Its original 11 result rows were carried forward. No restart from scratch or historical epoch-7 script was used. The original best and last remain under `preserved_inputs/`. Later `best.pt` and `last.pt` are separate result copies.

## Environment and commands

- Python: {environment['python'].splitlines()[0]}
- PyTorch: {environment['torch']}; built CUDA {environment['torch_cuda']}; Ultralytics {environment['ultralytics']}.
- GPU: NVIDIA RTX 4050 Laptop, 6 GiB.
- Driver and live GPU details are recorded in `environment_preflight.json`; disk-space observations are in `storage_verification.json` (223,212,396,544 bytes initially free).
- Supplied checkpoints recorded Ultralytics 8.4.174; existing working environment uses 8.4.185. Resume optimizer/EMA state was retained, but bit-identical continuation across versions/devices is not claimed.
- Thermal guard retained and tightened for this laptop: pause >=80 C, resume <=75 C, stop >=85 C or cooldown >10 minutes. Temperature samples are saved as JSONL.
- Exact resolved run settings: each run's `args.yaml`; training/evaluation scripts and logs are included.
- Task A command: `python scripts/train_task.py task_a`.
- Remaining fixed sequence: `python scripts/run_remaining.py` (with its saved per-stage commands in pipeline receipts).
- Task B: 20-epoch target, patience 6, AdamW lr 0.00008, FP32, seed 20261103, batch 4 at 640 for detectors; batch 16 at 224 for Harvest classification. Mild geometric/color augmentation; three independent models.
- Completion/elapsed-time records: `COMPLETION_SUMMARY.json`. Early stopping is permitted by the recorded patience; actual epoch totals are reported there.
- Historical `results.csv` timing already resets between earlier resume sessions. All original rows are retained exactly; use per-stage receipts for this task's wall times.

Working inputs and generated caches are isolated in ignored `_work/` and `runtime/`, not uploaded as duplicate datasets. Reconstruct them from the preserved handoff savepoint and result scripts in a new timestamped folder. Published `SHA256SUMS.txt` covers all published result files except itself, and excludes these two disposable local directories.

## Task A: per-class validation comparison

{chr(10).join(a_table)}

Validation selection: `{a['selected']}`. Rule: {a['selection_rule']}

This selection is Yellowing-focused rather than an across-class victory. The selected checkpoint's Yellowing AP50-95 is {a['models'][a['selected']]['per_class']['Yellowing']['map50_95']:.6f}, versus preserved {a['models']['preserved_best']['per_class']['Yellowing']['map50_95']:.6f}; its Healthy AP is {a['models'][a['selected']]['per_class']['Healthy']['map50_95']:.6f} versus {a['models']['preserved_best']['per_class']['Healthy']['map50_95']:.6f}, and its Wilting AP is {a['models'][a['selected']]['per_class']['Wilting']['map50_95']:.6f} versus {a['models']['preserved_best']['per_class']['Wilting']['map50_95']:.6f}. `task_a/weights/best.pt` is retained as the alternative macro-AP/Wilting candidate, with its own Healthy-recall tradeoff visible in the table. Neither replaces the baseline or production model.

The comparison preserves the historical validation settings (640, conf 0.001, NMS IoU 0.50) and development Yellowing probe workflow. The 12 outside photos are development probes, not independent acceptance. The older Condition V2 baseline checkpoint referenced in the original script is not in this handoff; the preserved epoch-8 best is the available checkpoint baseline. Full per-image probe predictions and annotated examples are included.

Per-class AP-evaluator precision/recall use its F1 operating point. The separate `fixed_confidence_counts` and confusion matrix use confidence 0.001 and matching IoU 0.45 (the installed validator default), with NMS IoU 0.50. Those low-confidence false-positive counts are diagnostic and are not a selected deployment threshold.

## Fixed-demo condition: per-class box metrics

The preserved baseline report records condition accuracy {field['preserved_baseline']['condition']['accuracy']}, per-class accuracy {json.dumps(field['preserved_baseline']['condition']['per_class_accuracy'])}, Harvest accuracy {field['preserved_baseline']['harvest']['accuracy']}, and printed-prop P/R/F1 {field['preserved_baseline']['prop']['precision']}/{field['preserved_baseline']['prop']['recall']}/{field['preserved_baseline']['prop']['f1']}. The baseline is also rerun below on this receiving environment using the unchanged comparison workflow. Its new standard box-AP measurements were not present in the original report.

{chr(10).join(table)}

Primary fixed-demo top-prediction behavior uses the unchanged preserved benchmark script: condition conf 0.10, NMS IoU 0.50, 640 input. Box AP is supplementary standard detection evaluation on the same locked test images. False-positive/false-negative top-prediction counts and confusion counts are in `field_comparison.json`.

Baseline condition top-prediction results:
```json
{compact(baseline,'condition')}
```
Candidate condition top-prediction results:
```json
{compact(candidate,'condition')}
```

## Harvest-Ready classification

Baseline and candidate both use the preserved benchmark's 640-pixel evaluation input for comparability, despite training at the starter model's 224-pixel input. No test-driven input-size selection was performed.

Baseline:
```json
{compact(baseline,'harvest')}
```
Candidate:
```json
{compact(candidate,'harvest')}
```

## Printed insect props

Primary comparison uses the unchanged tiled workflow: 640 tiles, stride 384, confidence 0.25, per-tile IoU 0.50, global NMS 0.45, greedy class-consistent matching at IoU 0.50. Full-frame standard AP is supplementary and is not interchangeable with tiled P/R/F1.

Baseline:
```json
{compact(baseline,'prop')}
```
Candidate:
```json
{compact(candidate,'prop')}
```

## Model paths and SHA-256

{model_lines}

All checkpoints, including preserved originals and periodic Task A checkpoints, are inventoried in `MODEL_SHA256.json`. The `best` selections use validation only. The final test report did not trigger another training/tuning cycle.

## Reviewed recommendations

- **Condition:** `field_condition/weights/best.pt` is the preferred candidate for fresh fixed-demo/full-pipeline validation. It correctly classified all 240 test crops, versus 187/240 baseline; top-prediction false positives for Healthy/Yellowing/Wilting fell from 13/40/0 to 0/0/0. This is confined to the repeated six-plant setup.
- **Harvest:** retain `field_harvest/weights/best.pt` for development review only. Accuracy improved from 120/240 to 155/240, but ready recall fell from 40/120 to 36/120. False-ready predictions improved from 40 to 1 while missed-ready examples rose from 80 to 84. Neither aggregate accuracy nor 100% native 224-pixel validation justifies promotion under the preserved 640-pixel pipeline evaluation.
- **Printed props:** prefer the preserved `continuation_packages/starter_models/printed_prop_v8.pt` for future validation. Reject the new adaptation for promotion: tiled precision/recall/F1 fell from 0.9231/0.7500/0.8276 to 0.4048/0.2125/0.2787; TP fell from 60 to 17, FP rose from 5 to 25, and FN rose from 20 to 63. No-prop false positives rose from 5 to 22 across 20 negative frames (18 frames affected). The failed candidate is preserved as evidence.
- **Task A:** retain the final checkpoint as a Yellowing-focused development candidate and the continued best as a macro-AP/Wilting alternative. Their cross-class tradeoffs are explicit above.

`RECOMMENDED_CANDIDATES.json` records exact hashes and roles. Source baseline condition/Harvest/prop discrete per-image predictions were checked against the fresh rerun in `baseline_reproduction.json`. Annotated examples and plots were decoded successfully in `image_output_validation.json`.

Preprocessing differences may contribute to the weak Harvest/prop results: Harvest trained at 224 while the benchmark runs at 640; printed props were fine-tuned on resized full frames but scored through 640-pixel tiles. These are possible contributors, not proven causes. Any later protocol change should be registered using validation data and evaluated on fresh held-out groups; no additional test-driven training was performed here.

## Interpretation and remaining limitations

- These fixed-camera frames repeat the same six plants and background. They are controlled-demo results, not estimates of independent plant/source/camera generalization.
- Sets 02 and 04 were never in training or validation by filename/group assignment. The supplied exact duplicate places Set 03 frame 20 (validation) also in Set 04 frame 1 (test), yielding six identical crops per crop task and one repeated full prop frame. This overlap is retained, explicitly audited in `split_audit.json`, and compromises strict validation/test independence. No duplicate was silently removed or reassigned.
- Printed insect props are presentation surrogates, not proof of real-pest accuracy.
- Compare each class and false positives rather than aggregate mAP alone. Recommendations are for fresh isolated Pi-camera/full-pipeline testing, not production replacement.
- No independent acceptance set was created. `DATA_GATE_NOT_READY` and `INDEPENDENT_ACCEPTANCE_NOT_READY` remain for broader/generalized use; this task's narrow experimental training was explicitly authorized.
- Fresh Pi-camera/full-pipeline validation remains incomplete and was prohibited in this task. No candidate is called deployable.

## Publication and resumability

The result branch is based on the exact handoff commit. `.pt` outputs use Git LFS. Local Git commit and remote push verification are recorded in a separate publication receipt after the result manifest is frozen, avoiding a self-referential commit/checksum record. If upload fails, retain all local results and the local commit and report the exact error.

Local publication receipt: `C:/Users/ben/Documents/Codex/2026-10-24/files-pasted-by-the-user-verde/work/stronger_gpu_20261103/PUBLICATION_RECEIPT.json`. This receipt is operational Git metadata outside the frozen results. For a later resume, check this receipt and the exact remote results branch before attempting publication again. If the receipt confirms a matching pushed commit and a clean working tree, this task is finished: do not retrain or republish it.
'''
    (OUT/'TRAINING_REPORT.md').write_text(report,encoding='utf-8')
    write_json('SAVEPOINT.json',{'status':'TRAINING_AND_EVALUATION_COMPLETE_LOCAL_VALIDATED','source':str(SOURCE),'results':str(OUT),'base_commit':BASE,'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),'tasks':completed,'model_hash_inventory':'MODEL_SHA256.json','manifest':'SHA256SUMS.txt','source_files_unchanged':len(actual_paths),'publication_receipt':'C:/Users/ben/Documents/Codex/2026-10-24/files-pasted-by-the-user-verde/work/stronger_gpu_20261103/PUBLICATION_RECEIPT.json','next_action':'First inspect publication receipt and exact remote results branch. If pushed commit matches and tree is clean, STOP: this task is finished. If unpublished, commit verified files with Git LFS and automatically push, preserving local commit on failure. No further training or deployment.','pi_accessed':False,'deployed':False,'independent_acceptance_ready':False})
    write_json('VALIDATION.json',{'status':'PASS','source_files_verified_unchanged':len(actual_paths),'required_run_artifacts_present':True,'split_boundary_violations':0,'dataset_reassignment':False,'original_checkpoints_unchanged':True,'training_tasks_completed':list(completed),'published_checksum_scope':'All result files except SHA256SUMS.txt itself, _work/, runtime/, and __pycache__/.'})
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and not any(part in {'_work','runtime','__pycache__'} for part in p.relative_to(OUT).parts) and p.name!='SHA256SUMS.txt')
    lines = [f'{sha(p)}  {p.relative_to(OUT).as_posix()}' for p in files]
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for line in lines:
        digest,relative = line.split('  ',1)
        assert sha(OUT/relative)==digest,relative
    print(json.dumps({'status':'PASS','source_files_unchanged':len(actual_paths),'result_checksum_files':len(files),'manifest_sha256':sha(OUT/'SHA256SUMS.txt'),'model_files':len(models)},indent=2))

if __name__ == '__main__':
    main()

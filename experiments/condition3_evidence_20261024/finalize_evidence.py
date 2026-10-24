"""Write evidence-backed status and verify preservation of protected local inputs."""
from __future__ import annotations
import csv,hashlib,importlib.metadata,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent;E=HERE/'evidence';REPO=HERE.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields=None):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()

def main():
    audit=json.loads((E/'INTAKE_AUDIT.json').read_text());visual=json.loads((E/'VISUAL_REVIEW_SUMMARY.json').read_text())
    runtime=json.loads((E/'RUNTIME.json').read_text());runtime.pop('seed',None);runtime.update({'seed_override':None,'augmentation':False,'cpu':'Intel Core i3-9100 @ 3.60GHz, four cores','gpu':'Intel UHD Graphics 630; no NVIDIA/CUDA device','ram_bytes':17049088000,'faster_coco_eval':importlib.metadata.version('faster-coco-eval'),'dependency_note':'Ultralytics auto-installed faster-coco-eval 1.8.0 for supplemental COCO evaluation. Primary reported metrics remain Ultralytics results.'})
    (E/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n',encoding='utf-8')
    distributions=sorted([{'package':d.metadata.get('Name','UNKNOWN'),'version':d.version} for d in importlib.metadata.distributions()],key=lambda r:(r['package'].lower(),r['version']))
    (E/'INSTALLED_DISTRIBUTIONS.json').write_text(json.dumps(distributions,indent=2)+'\n',encoding='utf-8')
    reports=[]
    for stem in ['verde_condition3_best_validation','verde_condition3_final_gpu_resume']:
        p=E/(stem+'_cpu_retest.json');r=json.loads(p.read_text());r['runtime']=runtime;p.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');reports.append(r)
    print('Checking preservation of',audit['protected_files'],'files',flush=True)
    before=read(E/'PROTECTED_INPUT_CHECKSUMS_BEFORE.csv');checks=[]
    for r in before:
        p=Path(r['path']);actual=sha(p) if p.is_file() else 'MISSING';checks.append({'path':str(p),'before_sha256':r['sha256'],'after_sha256':actual,'match':actual==r['sha256']})
    write(E/'PROTECTED_INPUT_VERIFICATION_AFTER.csv',checks)
    failures=[r for r in checks if not r['match']]
    if failures:raise RuntimeError('Protected files changed: '+repr(failures[:5]))
    print('Protected inputs unchanged:',len(checks),flush=True)
    fixed=read(E/'FIXED_CONFIDENCE_CLASS_METRICS.csv')
    model_rows=[]
    for report in reports:
        stem=Path(report['checkpoint']).stem
        for c in report['per_class']:
            f=next(r for r in fixed if r['checkpoint']==stem and r['class']==c['class'])
            model_rows.append({'checkpoint':stem,'sha256':report['sha256'],'class':c['class'],'internal_AP50':c['ap50'],'internal_AP50_95':c['ap50_95'],'precision_conf025':f['precision'],'recall_conf025':f['recall'],'f1_conf025':f['f1'],'FP_conf025':f['false_positive'],'FN_conf025':f['false_negative'],'independent_AP50':'NOT_AVAILABLE','deployment_decision':'DO_NOT_DEPLOY'})
    write(E/'MODEL_COMPARISON.csv',model_rows)
    table='\n'.join(f"| {r['checkpoint'].replace('verde_condition3_','')} | {r['class']} | {100*float(r['precision_conf025']):.2f}% | {100*float(r['recall_conf025']):.2f}% | {100*float(r['f1_conf025']):.2f}% | {100*float(r['internal_AP50']):.2f}% | {100*float(r['internal_AP50_95']):.2f}% |" for r in model_rows)
    starting=git('rev-parse','HEAD');branch=git('branch','--show-current')
    status=f'''# VERDE October 24 evidence status

Decision: **DO NOT DEPLOY. Preserve the validation-selected experimental rollback.** No new model was trained. The existing production Raspberry Pi model was not accessed or changed.

## Actual work completed

- Fetched origin; main remains at `9c410c435d2ed5e883da3e64b92167fd74ee7593`. Created isolated branch `{branch}` at `{REPO}`.
- Verified this PC: Intel i3-9100, 16 GB RAM, Intel UHD 630, PyTorch {runtime['torch']}, CUDA unavailable. Runtime package details are saved separately. This machine is not the collaborator's GTX 1650 device.
- Saved rollback copies of both experiment checkpoints under `{REPO / 'runs' / 'rollback_20261024'}` and verified their SHA-256 values. The checkpoint copies are local; no duplicate weights are added to Git.
- Preserved and rechecked **{len(checks):,} files**, covering the restored image/annotation sources, canonical annotation working set, canonical V2 archive if present, and the repository model files. All hashes match the before-state.
- Reproduced the existing three-class view: **1,167 images / 1,282 boxes**; Healthy 233, Yellowing 549, Wilting 385. Train 760, validation 207, test 200. Copies use their own labels/cache paths; source labels/images remain unchanged.
- Verified {audit['groups']} declared groups, zero declared-group split overlaps and zero exact binary overlaps across the historical splits. **Six source families span splits.** One inferred D1 filename/date sequence family spans train and test; this is a review flag, not a confirmed identity claim. Group metadata alone does not prove source/plant independence.
- Recorded the five already-known D1 historical JPEG/canonical hash differences. Baseline retesting uses the historical binaries and labels, with their actual hashes. No canonical transfers were performed in this task.
- Visually inspected an 18-image stratified historical sample and saved review flags without changing labels. It includes leaf-dominated crops, chlorosis-versus-wilt ambiguity and growing-context concerns. This is Codex review only.
- Downloaded **26 original files** from the CC BY 4.0 Fevgas water-stress dataset; 26/26 match publisher SHA-256. Extracted 26 embedded RGB JPEG streams losslessly and recorded byte linkage. The entire source has **two plants/groups**, not 26 independent plants. All remain **HOLD_GROWING_PLANT_CONTEXT**; zero are added to training. Source treatment was not assigned as a visual condition label. No pilot inference or annotation boxes were generated.

## Measured internal results

Both checkpoints were evaluated once on the same 200-image historical test split, 217 boxes, image size 416, CPU, batch 4, AP confidence floor 0.001, NMS IoU 0.7 and max detections 300. The evaluation commands and log are preserved. The installed runtime differs from the repository's pinned Ultralytics 8.4.174, but reproduced the recorded primary aggregate/AP results.

| Checkpoint | Class | Precision conf=.25 | Recall conf=.25 | F1 conf=.25 | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|---:|
{table}

Aggregate validation-selected: mAP50 **{100*reports[0]['metrics']['metrics/mAP50(B)']:.2f}%**, mAP50:95 **{100*reports[0]['metrics']['metrics/mAP50-95(B)']:.2f}%**. Aggregate final GPU-resumed: **{100*reports[1]['metrics']['metrics/mAP50(B)']:.2f}%**, **{100*reports[1]['metrics']['metrics/mAP50-95(B)']:.2f}%**.

Fixed-threshold precision/recall/F1 above use class-aware greedy IoU>=0.5 matching at conf>=0.25 against unchanged historical labels. AP uses the confidence sweep. Ultralytics' own P/R/F1 from its best mean-F1 point on this test curve are separately preserved and are descriptive, not a threshold selected on validation for deployment.

### Observed failures and benchmark limitations

- `D1-V006_77_1_10_05_03_12_1_2761e_20201209_47.jpeg` is historically Yellowing but validation-selected predicts Healthy at approximately 0.91. It is a leaf-dominated crop, so the example is not a complete growing-plant acceptance case.
- `B2-AGRIBOT-2025-04-30-126` has a historical box covering two distinguishable plants. Predictions separating those plants can score as extra boxes. Fixed-threshold FP/FN are disagreements with historical labels, not all confirmed biological failures. This label/instance issue must be reviewed before interpreting plant-separation metrics as final quality evidence.
- Final GPU-resumed Healthy fixed-threshold precision is lower despite higher aggregate AP50. That is another reason not to select a model solely by aggregate AP.
- Wilting AP50:95 remains approximately 47% for both checkpoints. Example boxes can include pot/soil or shifted foliage boundaries despite passing IoU=.5.
- Four earlier internet challenges and the five-photo PC check remain development evidence. Their original images and the six-epoch candidate are not in this fetched checkout. Their reported failures were not erased or relabeled as new measurements.

Annotated comparisons, class curves and confusion matrices are under `{E / 'error_review'}`. Machine-readable per-image errors, confidence values and hashes are included.

## New training and independent-test status

**New training approvals: 0. New training epochs: 0. Qualified untouched three-class acceptance images: 0. Independent per-class metrics: unavailable.**

The new source is held because planted/rooted context is not established, most early treatment frames do not establish visual wilting, and it contains no established Yellowing examples. Other leads are documented with access/rights limits. The repository describing 1,086 nutrient-deficiency originals does not distribute its dataset; its Apache code license is not evidence of image rights. No broad repeat search or unproductive unchanged-data fine-tune was used to manufacture a new checkpoint.

The >=85% mAP50 thesis target is a target. Existing internal scores exceed 85% on their historical benchmark but **do not establish independent acceptance or deployment readiness**.

## Rollback and exact continuation

- Retained validation-selected checkpoint: `{REPO / 'models' / 'verde_condition3_best_validation.pt'}`.
- SHA-256: `{reports[0]['sha256']}`.
- Final GPU-resumed comparison SHA-256: `{reports[1]['sha256']}`.
- No new best.pt/last.pt or training command exists for this task because training did not start.
- Evaluation command: `python experiments/condition3_evidence_20261024/evaluate_preserved_models.py`.
- New-data source/rights/assignments: `evidence/new_source_pilot/CODEX_VISUAL_REVIEW.csv` and `EMBEDDED_RGB_LINKAGE.csv`.
- Acceptance and live Pi requirements: `evidence/ACCEPTANCE_AND_PI_GATE.md`.
- Local savepoint: `{E / 'SAVEPOINT.json'}`.
- Isolated branch: `{branch}`. The final response/ignored local push receipt records the evidence commit after push.

Continue by obtaining clearly documented growing-plant symptom data and an all-three-class source/plant-disjoint acceptance set, reviewing ambiguous historical instance labels in a separate corrected dataset, and running the next approved-data candidate on verified GPU hardware. Live Pi Camera Module 3 capture, timing/memory and rollback tests are still required. No Pi deployment package was prepared because independent acceptance did not pass. Unknown remains abstention; Harvest-Ready is a separate manual/date observation; Pest is a separate unvalidated task.
'''
    (E/'FINAL_STATUS.md').write_text(status,encoding='utf-8')
    save={'created_utc':datetime.now(timezone.utc).isoformat(),'phase':'EVIDENCE_AUDIT_AND_INTERNAL_RETEST_COMPLETE','branch':branch,'starting_commit':starting,'workspace':str(REPO),'training_started':False,'new_training_approvals':0,'new_source_originals':26,'new_source_embedded_rgb':26,'new_source_groups':2,'new_source_holds':26,'qualified_independent_acceptance_images':0,'deployment_decision':'DO_NOT_DEPLOY','rollback_model':str(REPO/'models'/'verde_condition3_best_validation.pt'),'rollback_sha256':reports[0]['sha256'],'protected_files_unchanged':len(checks),'canonical_or_production_modified':False,'next_action':'Obtain and independently review whole growing-plant symptom images; freeze disjoint acceptance groups; use GPU for justified isolated candidate','final_report':str(E/'FINAL_STATUS.md')}
    (E/'SAVEPOINT.json').write_text(json.dumps(save,indent=2)+'\n',encoding='utf-8')
    # Checksum the reviewable evidence/scripts. Original and decoded binaries are
    # separately linked by hash in the pilot inventory and remain under ignored runs/.
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256' and '__pycache__' not in p.parts)
    (HERE/'CHECKSUMS.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(HERE).as_posix()}\n' for p in files),encoding='utf-8')
    print(json.dumps({'protected_unchanged':len(checks),'evidence_files':len(files),'report':str(E/'FINAL_STATUS.md'),'savepoint':str(E/'SAVEPOINT.json')},indent=2))

if __name__=='__main__':main()

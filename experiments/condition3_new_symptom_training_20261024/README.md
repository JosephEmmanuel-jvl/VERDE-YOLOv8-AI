# Experimental Condition3 symptom training handoff — 2026-10-24 evidence

**EXPERIMENTAL · NOT CANONICAL · NOT DEPLOYMENT READY · DO_NOT_DEPLOY**

`DATA_GATE_NOT_READY` · `INDEPENDENT_ACCEPTANCE_NOT_READY`

Purpose: document an isolated three-condition YOLO experiment with limited externally sourced,
human-reviewed Yellowing and Wilting evidence. No model has been trained from these additions.
Future checkpoints must be labeled `EXPERIMENTAL_CONDITION3_CANDIDATE`.

## Evidence and proposed build

Classes: **0 Healthy, 1 Yellowing, 2 Wilting**. Harvest-Ready is outside this condition model.
Pest is separate. Unknown is inference abstention, not a trained class.

| Component | Images/scenes | Approved boxes | New conservative groups |
|---|---:|---:|---:|
| Historical Healthy / Yellowing / Wilting | 233 / 549 / 385 = 1,167 | 1,282 existing boxes | Independence unproven |
| New LettuceDeF Yellowing | 6 | 6 | 1 |
| New LAB Wilting | 6 scenes | 36 | 1 |
| Proposed combined experiment | 1,179 | 1,324 | Only 2 new groups |

Historical split remains **760 train / 207 validation / 200 test**. Proposed augmentation:
**772 train / 207 validation / 200 historical test = 1,179**. All 12 additions are
`EXPERIMENTAL_TRAIN_CANDIDATE` (original approval-ledger role `TRAIN_CANDIDATE`).
Human phenotype approval and human box approval are separately recorded. Neither authorizes training now.

Whole new groups are assigned exclusively to training:

* `LETTUCEDEF_GREENHOUSE_COHORT_20250410_20250605`
* `LAB_SHARED_EXPERIMENT_20251125_20251210`

Six Yellowing images are one group; 36 Wilting boxes/appearances are not 36 independent plants.
The original 200 test image/label pairs are hash-frozen and byte-preserved. No exact historical
test-image duplicate enters training. Perceptual duplicates and historical plant/source/date
independence are not proven. This benchmark is **not** the final deployment acceptance set.
No independent Healthy/Yellowing/Wilting acceptance set exists; acceptance groups remain zero.
Only one new group per symptom exists, and project-owned collection has zero photographs.

## Repository representation and immutable sources

`handoff/` is a **sparse documentation mirror**, not the complete transfer workspace. It contains
every non-historical payload file except the existing baseline checkpoint, including all 12 new
original JPEGs, approved YOLO/AnyLabeling annotations, provenance ledger, manifests, config,
package version, original checksum manifest, scripts and savepoint. These files are byte-identical
to the validated handoff. Original source ledgers and older reports are preserved as dated audit
history; their earlier workflow states do not override current human approval or this publication.

The 1,167 historical image/label pairs and baseline are deliberately reused from existing
repository/LFS restoration; no archive or checkpoint is replaced. The root `SHA256SUMS.txt`
validates this repository experiment. `handoff/SHA256SUMS.txt` is the **full offline package's**
2,404-entry reference manifest; 2,334 historical files and one baseline are intentionally omitted
from this sparse mirror. Do not run the full transfer verifier directly in the sparse mirror.

The original transfer artifact remains offline:

* Filename: `VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1.zip`
* Bytes: **2,526,993,056** (about 2.35 GiB)
* SHA-256: `f13c14179ee732714641e0b4b8715aebac1a8a6247ebd96662819f49f8aa1a38`
* Original validation: 2,404 payload checksums / 2,405 ZIP-member CRC+SHA checks,
  fresh extraction, annotations and publisher GPU PASS; reverified before publication.

The ZIP duplicates historical data already available through four existing LFS archive parts.
No established policy requires a second copy, so this task uploads **no transfer ZIP, no new LFS
objects, and no replacement historical data**. The 12 source JPEGs total about 6.3 MB and are
ordinary Git blobs; small text annotations/metadata stay ordinary Git. No LFS quota change is needed.
Existing `.gitattributes`, historical LFS pointers and archive objects are unchanged.

## First action on the second training laptop

Using an authenticated GitHub client, clone the experimental branch into a NEW short path:

```powershell
git clone --branch experiment/condition3-new-symptom-training-20261024 --single-branch https://github.com/JosephEmmanuel-jvl/VERDE-YOLOv8-AI.git C:\VERDE_REVIEW
python -B C:\VERDE_REVIEW\experiments\condition3_new_symptom_training_20261024\scripts\verify_experiment.py
```

Use an existing working Python environment with Pillow. Do not reinstall working CUDA/PyTorch.
This first verifier checks committed hashes, approvals, images/annotations, grouping, counts,
test hash definitions and the repository baseline. It does **not** claim GPU or restored-history
verification unless those are separately performed.

If the original offline ZIP is available, copy it beside `transfer/VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1_START_HERE.ps1`
and run the helper from that transfer folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1_START_HERE.ps1 -Python python
```

The helper checks the exact ZIP SHA before extraction, extracts to a NEW short folder under
the receiving user's `VERDE_TRANSFERS`, and runs the full verifier including CUDA/PyTorch/
Ultralytics, baseline loading and a tiny CUDA arithmetic smoke test. It never installs or trains.
The publisher GPU result is not a receiving-laptop GPU pass.

## Reconstruct without the original PC or transfer ZIP

Follow the existing root [DATASET_TRANSFER.md](../../DATASET_TRANSFER.md). Restore the existing
LFS archive parts into a NEW local data directory; leave any existing canonical data untouched.
Prepare the historical three-class split into a NEW isolated path using the repository script.
For example, after restoring `C:\VERDE_DATA\verde_v2` and `verde_v2_anylabeling`:

```powershell
python src\prepare_dataset.py --source C:\VERDE_DATA\verde_v2 --decoded C:\VERDE_DATA\verde_v2_anylabeling --dest C:\VERDE_WORK\historical_three_class
python -B experiments\condition3_new_symptom_training_20261024\scripts\verify_experiment.py --historical C:\VERDE_WORK\historical_three_class
python -B experiments\condition3_new_symptom_training_20261024\scripts\reconstruct_handoff.py --historical C:\VERDE_WORK\historical_three_class --destination-parent C:\VERDE_HANDOFF
```

The destination parent must already exist; its versioned child must not. Reconstruction only
copies into that NEW directory outside the repository, checks every input/output hash, and runs
the exact full transfer verifier including GPU checks. `--plan-only` writes nothing;
`--package-only` explicitly skips GPU verification and is not a GPU pass.
The reconstructed **payload bytes** match all 2,404 original checksums; this does not recreate
the original ZIP's container timestamps/compression or promise its original archive SHA.
All scripts use current local inputs; old absolute paths in archival provenance are not dependencies.

## Configuration and future training instructions

The validated [training template](handoff/configs/training_template.yaml) is copied unchanged.
Recorded values: baseline checkpoint, epochs 50, patience 10, image size 640, fixed batch 4,
AMP true, device 0, workers 0, seed 20261023, deterministic true, cache false and resume false.
Optimizer is **not explicitly specified** in that validated template; its resolved Ultralytics
value is deferred to the training laptop and must be recorded before any later authorized run.
Output/model/data paths remain explicit placeholders. No new parameter was invented or changed.

For a later separately authorized controlled experiment, use identical initial checkpoint,
seed, resolved settings and evaluation policy for historical-control and augmented arms.
The included optional full-package builder refuses existing destinations and never trains.
Do not repeatedly tune against the historical test set. No training command is executed or
authorized by this publication, and closed data/acceptance gates remain closed.

Baseline/rollback: `models/verde_condition3_best_validation.pt`, SHA-256
`54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987`.
It is referenced from the existing repository, never overwritten or renamed as a new model.

## Approval caveats, rights and exclusions

All 42 retained boxes are HUMAN_BOX_APPROVED; all 12 classes are HUMAN_CLASS_APPROVED.
Human approval supersedes previous Codex workflow holds. No box correction was repeated.
Twenty genuine frame-clipped appearances (2 Yellowing, 18 Wilting) and three intentionally
unboxed LAB ambiguous edge regions are preserved. The full-scene experimental builder requires
explicit partial-scene risk acknowledgment; no ignore mask, hidden plant identity or annotation
was fabricated. Stock YOLO may treat unlabeled fragments as background; document a reviewed
policy before any future full-scene training. This caveat does not revoke retained-box approval.

See [ATTRIBUTION.md](ATTRIBUTION.md) and `rights/` for verified CC BY 4.0 image redistribution
evidence, creator credit, source links and annotation-change notices. These scientific status
warnings do not impose additional restrictions on CC BY image rights.

Excluded training evidence: LETTUCEDEF_011; NPK DEVELOPMENT_ONLY; Concepcion HOLD; rejected or
unresolved public candidates; 138-image WIP; planned project-owned plants; VERDE-TR-H-001
(zero photos); nonexistent independent acceptance claims. Some identifiers remain in historical
provenance/exclusion records only. No excluded image or WIP annotation is added here.

Deployment remains blocked. Raspberry Pi production weights, canonical V2, historical data,
historical annotations/checkpoints, existing LFS archives and previous finalized packages
remain unchanged. This branch is for review; it is not merged into `main`.

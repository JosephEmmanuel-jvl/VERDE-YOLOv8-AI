# VERDE experimental training handoff — version 1, 2026-10-24

This self-contained package prepares a controlled experiment on a separate NVIDIA GPU laptop.
No training was performed during packaging. No access to the original PC is needed.
Future models remain `EXPERIMENTAL_CONDITION3_CANDIDATE`, NOT DEPLOYMENT-READY.
`DATA_GATE_NOT_READY`; `INDEPENDENT_ACCEPTANCE_NOT_READY`.

## What is included

* `historical/`: unchanged prepared images and YOLO labels, 1,167 images / 1,282 boxes:
  Healthy 233, Yellowing 549, Wilting 385. Original train 760 / val 207 / test 200.
* `new_symptom_evidence/`: copies of 12 approved originals, unchanged human-approved YOLO
  and AnyLabeling annotations, approval receipts, original provenance and approval ledger.
  Six Yellowing images / 6 boxes / ONE LettuceDeF group; six LAB Wilting scenes / 36 boxes /
  ONE LAB group. The 36 appearances are not 36 independent plants or samples.
* `models/verde_condition3_best_validation.pt`: verified actual baseline/rollback checkpoint.
  SHA-256 `54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987`.
* `manifests/`: individual image/label hashes, frozen historical test mapping, current
  experimental role ledger, two-group registry and proposed build plan.
* `provenance/historical_split_manifest.csv`: exact original split CSV; original data.yaml
  is retained ONLY as archival provenance. Its old absolute path is not used by any script.
* `configs/`: class map, seed and an inactive training configuration template.
* `scripts/verify_transfer.py`, `scripts/build_experiment.py`, `SAVEPOINT.json` and checksums.

No held/rejected/WIP image or annotation is included. Exclusions are recorded in
`manifests/exclusions.json`. There is no project-owned training evidence: VERDE-TR-H-001
has no photographs. Public sources were not reacquired or re-screened.

## First actions on the second GPU laptop

Copy the ZIP, sibling `START_HERE.ps1`, `.zip.sha256`, final report and validation receipt.
In PowerShell, change to that transfer folder, then run:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1_START_HERE.ps1 -Python python

Use `-Python 'C:\path\to\existing\venv\Scripts\python.exe'` if needed. The bootstrap first
compares the ZIP to its embedded expected SHA-256, refuses an existing extraction destination,
extracts into a NEW directory and runs the portable verifier. It never installs or trains.
The exact ZIP digest is in the sibling `.zip.sha256`, final report and bootstrap. It cannot
be embedded in the ZIP itself without creating a circular hash dependency.

The environment must already supply Python 3.12, Pillow, CUDA-enabled PyTorch and Ultralytics.
Known tested publisher environment: Python 3.12.10 / torch 2.9.0+cu128 / CUDA runtime 12.8 /
Ultralytics 8.4.185 / RTX 4050 Laptop GPU. A different NVIDIA GPU need not be an RTX 4050;
the verifier checks availability, reports VRAM, runs a tiny CUDA arithmetic smoke test,
and loads only the included, hash-verified baseline checkpoint. It does not train or install.
An absent/broken dependency is a failing verification, not an invitation to reinstall blindly.

Manual verification after extraction is also supported:

    python -B .\VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1\scripts\verify_transfer.py --zip .\VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1.zip --expected-zip-sha256 THE_DIGEST_FROM_THE_SIBLING_SHA256_FILE

Default verification covers ZIP SHA, every ZIP member CRC and SHA, exact extracted inventory,
internal checksums, original image hashes/readability, YOLO structure/classes/geometry,
AnyLabeling parity, approvals, group roles, baseline and CUDA environment. Any failure exits 1.
`--package-only` skips GPU checking and is not a GPU pass. `--internal-only` is publisher
staging mode and is not transfer-ZIP validation. Python is run with `-B` to avoid adding
unlisted __pycache__ files to the immutable package.

## Proposed experiment: no training authorization is conveyed

The original sources remain immutable and separate. `proposed_experiment_build.json` proposes:

* Historical control: train 760 / val 207 / test 200, total 1,167.
* Plus-new candidate: train 772 / val 207 / test 200, total 1,179 / 1,324 boxes.
  The 12 additions are all `EXPERIMENTAL_TRAIN_CANDIDATE` in training. Entire new groups
  stay together. Neither group contributes validation, test or independent acceptance data.
* The 200 historical test image AND label hashes are unchanged. Exact train/test and
  val/test hash overlap is zero; new/historical exact image overlap is also zero.
  This does not prove absence of perceptual derivatives or historical plant/source leakage.
  Historical plant/source/date group identity remains UNKNOWN, not invented per image.
* The historical test set is a comparison benchmark, NOT an independently proven final
  acceptance set. Do not repeatedly tune on it or call its result deployment validation.
* A controlled future comparison must use the same initial checkpoint, seed, resolved
  settings and evaluation policy for control and augmented arms. The included historical
  trained checkpoint is also a rollback/reference; comparing it alone with a differently
  trained new model is not a matched retraining experiment.

The optional builder only copies into a NEW explicit destination outside this package;
it refuses existing destinations, verifies sources first and verifies every copied file.
It writes portable data.yaml and a hash-linked BUILD_RECEIPT.json. It never calls YOLO.train.
Use `--variant historical-control` or `--variant plus-new`; both require the ZIP and its SHA.
Add `--plan-only` to inspect allocation without constructing a dataset. The training template
has deliberate path placeholders; resolve them only for a later separately authorized run.

## Annotation scope and unresolved LAB partial-scene risk

All 42 retained boxes are HUMAN_BOX_APPROVED; all 12 phenotypes are HUMAN_CLASS_APPROVED.
Human approval supersedes old Codex workflow QA holds. No new box correction was performed.
Twenty frame-clipped appearances are preserved (Yellowing 2, Wilting 18). Three intentionally
unboxed ambiguous edge regions remain unchanged: Cam5 upper-left, Cam17 upper-left, Cam22
right edge. Exact coordinates, reasons, flags and uncertainty are retained in the ledger.
Approval covers retained boxes; it does not identify these fragments or approve missing labels.

Stock YOLO has no ignore policy supplied here. Full-scene training may treat those fragments
as negative background. Accordingly the full 12-record builder requires the explicit
`--acknowledge-partial-scene-risk` flag. This records a controlled experimental limitation;
it does not resolve ambiguity, fabricate masks, change annotations, reopen human box approval,
or authorize substantive training. If that risk is unacceptable, stop and design a reviewed
ignore/exclusion policy before training; do not silently drop individual scenes and inflate
independence. The package contains the proposed full-cohort plan and all approved evidence.

## Provenance, rights and exclusions

The two public new-image source families are LettuceDeF (Zenodo record 21907611) and LAB-Lettuce
(exact original URLs/creators in the copied provenance ledger). Both were recorded as CC BY 4.0;
retain attribution, source links and modification/annotation history. Images/approved boxes
are copied byte-for-byte. Original absolute source paths are archival strings only; executable
scripts do not depend on them. Historical rights/provenance are limited to the existing
project records and are not expanded into a new blanket license claim by this handoff.

Excluded: LETTUCEDEF_011; NPK DEVELOPMENT_ONLY; Concepcion HOLD; experimental 138-image WIP;
planned project-owned plants; VERDE-TR-H-001; all acceptance claims. Original public source
archive provenance limitations remain as previously recorded; this package verifies the
included image/annotation bytes, not an unavailable original archive in its entirety.

No independent acceptance groups are available. Project-owned collection was not completed.
Neither a functioning GPU nor these two cohorts closes the data/acceptance gates.
Do not merge into canonical/historical data, overwrite baseline/Pi weights, deploy, commit or push.
Stop after transfer verification and experiment preparation until future work is authorized.

# Stronger-GPU Codex handoff

## Objective

Continue the paused Yellowing-focused condition training, then perform isolated fixed-demo adaptation for the three independent tasks:

1. Healthy / Yellowing / Wilting detection.
2. Harvest-Ready classification.
3. Printed insect-prop detection used as a presentation surrogate for pest detection.

Do not merge weights from separate YOLO tasks. Do not access, modify, or deploy to the Raspberry Pi. Preserve every supplied checkpoint and dataset unchanged.

## Initial setup

1. Check out this handoff branch and run `git lfs pull`.
2. Restore the archive into an empty local folder:

   ```powershell
   python continuation_packages/restore_continuation_savepoint.py --output-dir C:/VERDE-restore
   ```

3. Read the restored `STATUS.md`, `TRAINING_STATE.md`, `FULL_SHA256SUMS.txt`, dataset summary, baseline report, and training logs before starting.
4. Verify CUDA, PyTorch, Ultralytics, free disk space, and the copied checkpoint hashes. Record the environment in the final report.
5. Work only in a new timestamped folder under `stronger_gpu_results/`. Never write into the restored package.

## Task A: finish the paused condition run

- Dataset: restored `morning-yellowing-training/candidate_dataset`.
- Resume checkpoint: restored `morning-yellowing-training/weights/last.pt`.
- Resume-checkpoint SHA-256: `147c39ddc83a72e9b1d5e45e0cb6eab334ea07e5ad154fc08d3250f7c671f55d`.
- Preserved best checkpoint SHA-256: `83f2f0e9522a8c5e7258fcab7ccf4019205c8d549a956657d814b63adb1ff730`.
- Run settings: restored `morning-yellowing-training/run/args.yaml`.
- Recorded state: 11 completed result rows out of an intended 20 epochs. The best aggregate validation mAP50-95 row so far is epoch 8 at `0.81788`.

Resume through Ultralytics checkpoint state from `last.pt`. The included historical `resume_epoch8_20261103-123356.py` points to an older epoch-7 checkpoint and must not be run unchanged. Retain the existing thermal guard and stop safely before hardware thermal limits are exceeded.

After completion, evaluate the new `best.pt` and `last.pt` using the preserved comparison workflow. Report per-class precision, recall, mAP50, and mAP50-95 when available. Do not call the model better solely because aggregate mAP increased.

## Task B: fixed-demo field adaptation

Use only the restored `field-adaptation/dataset-v2` final dataset. Treat `dataset`, `baseline`, and other first-pass paths as superseded.

- Condition detection: `field-adaptation/dataset-v2/condition_detect/data.yaml`. Start from the best verified Task A candidate, while preserving it unchanged.
- Harvest classification: `field-adaptation/dataset-v2/harvest_classify`. Start from `continuation_packages/starter_models/harvest_cls_v3.pt`.
- Printed-prop detection: `field-adaptation/dataset-v2/prop_detect/data.yaml`. Start from `continuation_packages/starter_models/printed_prop_v8.pt`.

Train each task in its own isolated run directory. Preserve the split exactly:

- Training: first 15 frames from Sets 01 and 03.
- Validation: last 5 frames from Sets 01 and 03.
- Locked test: every frame from Set 02, where props were moved, and Set 04, which has no props.

Do not train on the locked test groups. The source contains one documented exact duplicate; do not silently remove or reassign it without documenting the change.

Compare every candidate against `field-adaptation/baseline-v2/baseline-report.json`. At minimum, report condition per-class results, Harvest-Ready test accuracy/confusion counts, and printed-prop precision/recall/F1 including false positives on no-prop frames. These results apply only to the fixed six-slot defense setup. Printed props are not evidence of real biological-pest performance.

## Required result package

Create `stronger_gpu_results/<YYYYMMDD-HHMM>/` containing:

- Task A final `best.pt`, `last.pt`, `args.yaml`, `results.csv`, logs, and comparison report.
- Each Task B run's `best.pt`, `last.pt`, `args.yaml`, `results.csv`, evaluation report, and representative annotated outputs.
- `TRAINING_REPORT.md` with environment, starting hashes, commands, elapsed time, per-class metrics, baseline comparison, limitations, and exact recommended candidates.
- `SHA256SUMS.txt` covering every result file.

Do not label a candidate deployable without an independent fresh Pi-camera/full-pipeline validation. Do not modify production or Pi files.

## Mandatory push after completion

The user explicitly requested that completed work be pushed automatically.

1. Fetch the remote and create a new results branch from this handoff branch, for example `training/stronger-gpu-continuation-<YYYYMMDD-HHMM>`.
2. Store all outputs only under `stronger_gpu_results/<timestamp>/`.
3. Ensure checkpoint files are handled by Git LFS. Add an LFS rule for `stronger_gpu_results/**/*.pt` if the rule is not already present.
4. Before committing, verify that staged `.pt` files are LFS pointer text with `git show :path/to/file.pt` and run `git lfs fsck`.
5. Commit with a message that states which tasks completed.
6. Push with:

   ```powershell
   git push -u origin HEAD
   ```

7. Verify the remote branch commit with `git ls-remote origin HEAD` or the exact branch reference and report the branch name, commit hash, model hashes, and any incomplete work.

If upload or quota fails, do not delete local results or retry destructively. Keep the local commit intact and report the exact Git/Git LFS error plus the resumable local path.

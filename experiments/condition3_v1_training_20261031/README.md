# EXPERIMENTAL CONDITION3-V1 TRAINING RESULT

**VERDICT: `CANDIDATE_MIXED`**

**NOT PRODUCTION · NOT PI-DEPLOYMENT READY · INDEPENDENT ACCEPTANCE NOT READY · `DO_NOT_DEPLOY`**
`CONDITION3-V2 NOT YET BUILT`

This folder archives one controlled GPU training run of the frozen Condition3-V1 dataset, with its evaluation. It is an experimental result only. The candidate is **not** the best, final, production, canonical or Pi model, and the preserved baseline stays the reference.

## What was trained
- **Source experiment (frozen, unmodified):** branch `experiment/condition3-new-symptom-training-20261024`, commit `2f2849c7de94df0b7a416c14eca6b9c40456332e`. Handoff ZIP `VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1.zip`, SHA-256 `f13c14179ee732714641e0b4b8715aebac1a8a6247ebd96662819f49f8aa1a38`.
- **Data:** 1,179 images: train 772 / validation 207 / test 200 (the 200-image historical test is unchanged). Classes `0 Healthy`, `1 Yellowing`, `2 Wilting`. The 12 new V1 records (6 LettuceDeF Yellowing, 6 LAB Wilting) are all in train. The dataset is **not** duplicated here: `dataset_manifest/BUILD_RECEIPT.json` lists every file with its hash.
- **Start model:** the preserved baseline `models/verde_condition3_best_validation.pt`, SHA-256 `54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987`. It was only read; it is unchanged.
- **Settings:** the frozen `training_template.yaml`, unchanged: 50 epochs max, patience 10, imgsz 640, batch 4, seed 20261023, deterministic, AMP, workers 0. `optimizer=auto` resolved to AdamW (lr 0.001429).
- **Environment:** RTX 4050 Laptop GPU (6 GB), PyTorch 2.9.0+cu128, Ultralytics 8.4.185, Python 3.12.10. Full versions are in `config/full_run_record.json`.
- **Result of the run:** early stop at epoch 27, **best epoch 17** (selected on validation), about **52.6 minutes**. One run only: no sweep, no repeats. The smoke test is recorded in `logs/smoke.log`; it is not a candidate.

| Checkpoint | SHA-256 |
|---|---|
| `weights/best.pt` (epoch 17) | `e6b3182c9edc67e51a65d660c0566122b85f2d0a9d173f6a133c81d55d40ffe9` |
| `weights/last.pt` (epoch 27) | `1b9142fd118e57f3bff6e77ecc418d5b6a73ceadad3fe5ac7ce170a2e4fba5d8` |

## Why the result is mixed
**Critical failure: the candidate did not solve new-source Yellowing.** On the four post-V1 development-challenge images (Coon g13-bg21, Coon g13-bg19, USDA N img05, USDA P img05), the candidate at its trained size of 640 detects Yellowing correctly on **0 / 4**. The baseline also scores 0 / 4. These images were discovered after V1 was frozen and were used **only** for inference; they were **not** in training and were not used to tune anything. They are a development challenge, not independent acceptance evidence. The model still has a Yellowing generalization / domain problem.

**What improved** (historical 200-image test, each model compared at the baseline's trained size 416):

| | Baseline @416 | Candidate @416 |
|---|--:|--:|
| Overall mAP50 / mAP50-95 | 95.00 / 69.99 | 96.87 / 78.71 |
| Healthy AP50-95 | 65.05 | 75.15 |
| Yellowing AP50-95 | 97.50 | 97.47 (no improvement) |
| Wilting AP50-95 | 47.43 | 63.51 |

- **Wilting:** box quality (AP50-95) is clearly better; recall and AP50 were already near ceiling.
- **Healthy:** AP and recall improved.
- **Yellowing:** no gain on the historical test; AP50 is slightly lower (99.32 to 98.50 at 416).

**What did not:**
- **Validation did not show a convincing overall improvement.** On the validation split, overall AP50 is lower at every comparison (about -1.5 at 640, -0.4 at 416, -0.7 each-at-trained-size) and AP50-95 is about flat to +1.3. Yellowing AP50 is lower in all three (-1.3 to -2.2). The large Wilting AP50-95 gain on test is much smaller on validation, so part of it is specific to the test set. Numbers: `reports/validation_split_metrics.json`.
- **More false positives and duplicates.** At conf 0.25 on test, the candidate has more Healthy false positives and duplicate boxes (candidate@640: 13 Healthy background FPs, 5 duplicate Healthy and 2 duplicate Wilting boxes, 2 merged plants) and a lower fixed-threshold F1 than the baseline at its own size (-3.69 points overall F1@.25). Several "background" FPs are real unlabeled neighbouring plants, a label limitation that also affects the baseline.
- **Do not reduce this to the test mAP gain.** See `reports/FINAL_EVALUATION_REPORT.md` for the full failure analysis.

## Image-size comparison caveat
The historical baseline was trained at imgsz **416**; the candidate at **640**. At 416 the baseline reproduces its recorded historical numbers; at 640 it degrades sharply (for example Wilting recall@.25 drops to 48.8%). The baseline-at-640 comparison therefore **must not** be read as the main evidence of improvement; it overstates the gain (+13.7 mAP50-95 vs +8.7 at 416). `reports/METRIC_TABLES.md` gives all three comparisons (640 vs 640, 416 vs 416, each at its trained size). The 416 evaluations were added as a control after the 640 results were seen; they selected nothing.

## TRAINING-DATA SANITY CHECK: NOT GENERALIZATION EVIDENCE
The 12 new V1 records are in training, so these results say nothing about generalization (candidate at 640):
- **LettuceDeF:** all 6 detected as Yellowing.
- **LAB:** all 36 annotated Wilting boxes detected, **but 52 predictions in total**: extra, spurious detections on single leaves and plumbing/frame-edge hardware. Some may fall on the three intentionally unboxed LAB edge regions.
- For contrast the baseline found 0 / 12 at 640 and 1 / 12 at 416.

## Test-set warning
The historical 200-image test set:
- has **unproven source/plant independence** (six source families are recorded as spanning historical splits);
- has now been **evaluated multiple times**;
- was evaluated at **multiple image sizes during this run**.

It must **not** be treated as untouched final acceptance evidence, and must **not** be used for future model-selection tuning. No independent acceptance set exists (`INDEPENDENT_ACCEPTANCE_NOT_READY`).

## Other limits
- **One arm only.** The handoff planned a matched historical-control vs plus-new comparison. This candidate also differs from the baseline in image size and training length, so its changes cannot be credited to the 12 new records.
- **Partial-scene risk accepted.** `--acknowledge-partial-scene-risk` was passed to build the authorized 1,179-image dataset; three unboxed LAB edge regions may have trained as background.
- **Small classes.** Healthy has 52 test boxes, so small deltas are noise.
- **Fixed-threshold P/R** for the baseline differs somewhat from the older recorded table even though AP matches; the earlier run used CPU and a different library set. Not investigated.

## Why the baseline remains preserved
The candidate is not better in a way that justifies replacing anything. The baseline `models/verde_condition3_best_validation.pt` is untouched and remains the rollback/reference. Nothing here touches Raspberry Pi weights, configuration, services, camera, backend, app, automation, dosing or SMS.

## Condition3-V2
**`CONDITION3-V2 NOT YET BUILT`.** The four post-V1 Yellowing images (published separately on branch `experiment/condition3-post-v1-yellowing-20261029`) belong to the future Condition3-V2 evidence queue and are not merged into this V1 result.

## Contents
- `reports/`: `FINAL_EVALUATION_REPORT.md` (start here), `METRIC_TABLES.md`, `comparison.json`, `validation_split_metrics.json`
- `weights/`: `best.pt`, `last.pt` (plain git files, matching how `models/*.pt` are stored in this repo; no LFS rule applies to them)
- `run/`: `results.csv`, `args.yaml`, training curves, confusion matrices, validation batches
- `config/`: exact training command and resolved run records (versions, GPU, seed, package freeze)
- `dataset_manifest/`: `BUILD_RECEIPT.json` (dataset/split manifest with hashes), `data.yaml`, `training_template.yaml`
- `eval/<model>_<imgsz>/`: `metrics.json`, `fixed_threshold_metrics.json`, `failures.json` and annotated `failures/`, `sanity_12*`, `post_v1*`, `val_test/` plots and confusion matrices; `eval/review_sheets/`
- `logs/`, `scripts/`, `RECOVERY.md`, `SOURCE.json`, `SHA256SUMS.txt`, `verify_package.py`

Not included on purpose: the 1,179-image dataset, per-epoch checkpoints (111 MB each), the smoke-test weights and the `yolo26n.pt` file that Ultralytics downloaded for its AMP check.

## Verify
```bash
python verify_package.py          # checksums, checkpoint hashes, best epoch, 0/4 challenge result
python verify_package.py --load   # additionally loads best.pt with Ultralytics (needs it installed)
```

Dataset attribution and rights stay as recorded in the frozen V1 handoff package.

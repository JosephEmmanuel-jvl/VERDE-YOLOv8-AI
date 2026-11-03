# VERDE stronger-GPU training continuation

Completed at 2026-11-03T21:43:44.833438+08:00 (Asia/Manila).

Base handoff: `7d7b81f19d01ee6f751f5a6a842626fec9a60f8f`. Results branch: `training/stronger-gpu-continuation-20261103-2013`.

## Scope and protection

All four authorized training tasks and their evaluations completed in this new result directory. The restored package's 4,917 files are unchanged in inventory and SHA-256. Original checkpoints and datasets were never used as output paths. No Raspberry Pi access, deployment, canonical/historical dataset modification, weight merging, or main-branch modification occurred.

Task A resumed checkpoint epoch 10 (zero-based), continuing with epoch 12 (one-based), all 183 optimizer-state entries restored. Its original 11 result rows were carried forward. No restart from scratch or historical epoch-7 script was used. The original best and last remain under `preserved_inputs/`. Later `best.pt` and `last.pt` are separate result copies.

## Environment and commands

- Python: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- PyTorch: 2.9.0+cu128; built CUDA 12.8; Ultralytics 8.4.185.
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

| Task A checkpoint | Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| preserved_best | Healthy | 0.896936 | 0.920000 | 0.936865 | 0.712422 |
| preserved_best | Yellowing | 1.000000 | 0.875963 | 0.953252 | 0.866894 |
| preserved_best | Wilting | 0.946859 | 1.000000 | 0.961509 | 0.885615 |
| continued_best | Healthy | 0.900090 | 0.900921 | 0.929207 | 0.721021 |
| continued_best | Yellowing | 1.000000 | 0.885754 | 0.943705 | 0.866908 |
| continued_best | Wilting | 0.950714 | 1.000000 | 0.970360 | 0.900814 |
| continued_last | Healthy | 0.898414 | 0.884460 | 0.932416 | 0.699087 |
| continued_last | Yellowing | 0.993894 | 0.878049 | 0.950819 | 0.875350 |
| continued_last | Wilting | 0.949918 | 1.000000 | 0.972830 | 0.883975 |

Validation selection: `continued_last`. Rule: Highest Yellowing validation AP50-95; Wilting AP50-95 tie-break. No fixed-demo test selection.

This selection is Yellowing-focused rather than an across-class victory. The selected checkpoint's Yellowing AP50-95 is 0.875350, versus preserved 0.866894; its Healthy AP is 0.699087 versus 0.712422, and its Wilting AP is 0.883975 versus 0.885615. `task_a/weights/best.pt` is retained as the alternative macro-AP/Wilting candidate, with its own Healthy-recall tradeoff visible in the table. Neither replaces the baseline or production model.

The comparison preserves the historical validation settings (640, conf 0.001, NMS IoU 0.50) and development Yellowing probe workflow. The 12 outside photos are development probes, not independent acceptance. The older Condition V2 baseline checkpoint referenced in the original script is not in this handoff; the preserved epoch-8 best is the available checkpoint baseline. Full per-image probe predictions and annotated examples are included.

Per-class AP-evaluator precision/recall use its F1 operating point. The separate `fixed_confidence_counts` and confusion matrix use confidence 0.001 and matching IoU 0.45 (the installed validator default), with NMS IoU 0.50. Those low-confidence false-positive counts are diagnostic and are not a selected deployment threshold.

## Fixed-demo condition: per-class box metrics

The preserved baseline report records condition accuracy 0.7792, per-class accuracy {"Yellowing": 0.675, "Healthy": 0.6667, "Wilting": 1.0}, Harvest accuracy 0.5, and printed-prop P/R/F1 0.9231/0.75/0.8276. The baseline is also rerun below on this receiving environment using the unchanged comparison workflow. Its new standard box-AP measurements were not present in the original report.

| Condition class | Baseline precision | Candidate precision | Baseline recall | Candidate recall | Baseline mAP50 | Candidate mAP50 | Baseline mAP50-95 | Candidate mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Healthy | 0.622762 | 0.999462 | 0.741667 | 1.000000 | 0.864388 | 0.995000 | 0.852083 | 0.990006 |
| Yellowing | 0.981474 | 0.998699 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.788145 | 0.995000 |
| Wilting | 0.988081 | 0.999241 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.556559 | 0.995000 |

Primary fixed-demo top-prediction behavior uses the unchanged preserved benchmark script: condition conf 0.10, NMS IoU 0.50, 640 input. Box AP is supplementary standard detection evaluation on the same locked test images. False-positive/false-negative top-prediction counts and confusion counts are in `field_comparison.json`.

Baseline condition top-prediction results:
```json
{
  "images": 240,
  "accuracy": 0.7792,
  "per_class_accuracy": {
    "Yellowing": 0.675,
    "Healthy": 0.6667,
    "Wilting": 1.0
  },
  "confusion": {
    "Healthy->Healthy": 80,
    "Healthy->Yellowing": 40,
    "Wilting->Wilting": 80,
    "Yellowing->Healthy": 13,
    "Yellowing->Yellowing": 27
  },
  "top_prediction_per_class_counts": {
    "Healthy": {
      "tp": 80,
      "fp": 13,
      "fn": 40,
      "precision": 0.8602150537634409,
      "recall": 0.6666666666666666
    },
    "Yellowing": {
      "tp": 27,
      "fp": 40,
      "fn": 13,
      "precision": 0.40298507462686567,
      "recall": 0.675
    },
    "Wilting": {
      "tp": 80,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
    }
  }
}
```
Candidate condition top-prediction results:
```json
{
  "images": 240,
  "accuracy": 1.0,
  "per_class_accuracy": {
    "Yellowing": 1.0,
    "Healthy": 1.0,
    "Wilting": 1.0
  },
  "confusion": {
    "Healthy->Healthy": 120,
    "Wilting->Wilting": 80,
    "Yellowing->Yellowing": 40
  },
  "top_prediction_per_class_counts": {
    "Healthy": {
      "tp": 120,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
    },
    "Yellowing": {
      "tp": 40,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
    },
    "Wilting": {
      "tp": 80,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
    }
  }
}
```

## Harvest-Ready classification

Baseline and candidate both use the preserved benchmark's 640-pixel evaluation input for comparability, despite training at the starter model's 224-pixel input. No test-driven input-size selection was performed.

Baseline:
```json
{
  "images": 240,
  "accuracy": 0.5,
  "confusion": {
    "not_ready->not_ready": 80,
    "not_ready->ready": 40,
    "ready->not_ready": 80,
    "ready->ready": 40
  }
}
```
Candidate:
```json
{
  "images": 240,
  "accuracy": 0.6458,
  "confusion": {
    "not_ready->not_ready": 119,
    "not_ready->ready": 1,
    "ready->not_ready": 84,
    "ready->ready": 36
  }
}
```

## Printed insect props

Primary comparison uses the unchanged tiled workflow: 640 tiles, stride 384, confidence 0.25, per-tile IoU 0.50, global NMS 0.45, greedy class-consistent matching at IoU 0.50. Full-frame standard AP is supplementary and is not interchangeable with tiled P/R/F1.

Baseline:
```json
{
  "images": 40,
  "tp": 60,
  "fp": 5,
  "fn": 20,
  "precision": 0.9231,
  "recall": 0.75,
  "f1": 0.8276,
  "no_prop_frames": 20,
  "no_prop_false_positive_boxes": 5,
  "no_prop_frames_with_false_positives": 5
}
```
Candidate:
```json
{
  "images": 40,
  "tp": 17,
  "fp": 25,
  "fn": 63,
  "precision": 0.4048,
  "recall": 0.2125,
  "f1": 0.2787,
  "no_prop_frames": 20,
  "no_prop_false_positive_boxes": 22,
  "no_prop_frames_with_false_positives": 18
}
```

## Model paths and SHA-256

- `field_condition/weights/best.pt` — `fbb0870e9a944ce03a66102597fc1e5c2bda453e0e5272583a3f3dd909b88212` (22,512,810 bytes)
- `field_condition/weights/last.pt` — `39a43c6ae67e6deed877cfdf342758c1a5e876e8ef4dc27041511370f1f62474` (22,512,810 bytes)
- `field_harvest/weights/best.pt` — `f6d3c5762b33706782d9396906fd408bc9706374b8c9b5d08ade39fda1189a06` (10,256,904 bytes)
- `field_harvest/weights/last.pt` — `eb7e9f00f93b6d212915952e55a45449a78700cafff4f681662fee338ec226f1` (10,256,904 bytes)
- `field_prop/weights/best.pt` — `51354e17320f15d428b0ebeb2a6aafaf2b3bfa43f4ac883383bd4e67045e27f2` (22,491,178 bytes)
- `field_prop/weights/last.pt` — `0f5cac71c59c1aea3c555704c52a3f2ff8c87d31de5bf128c41f666af160eff3` (22,491,178 bytes)
- `task_a/weights/best.pt` — `1eaf997a9706bd8a598acdd93dfbcf02037b77ad6d6e21ebdf02e1bae49a5998` (22,503,402 bytes)
- `task_a/weights/last.pt` — `c2b25b00aa678b60eed090fa6e0497d4801d490d0c458dd023fb9047c88b4dae` (22,503,402 bytes)

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

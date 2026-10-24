# VERDE October 24 evidence status

Decision: **DO NOT DEPLOY. Preserve the validation-selected experimental rollback.** No new model was trained. The existing production Raspberry Pi model was not accessed or changed.

## Actual work completed

- Fetched origin; main remains at `9c410c435d2ed5e883da3e64b92167fd74ee7593`. Created isolated branch `experiment/condition3-evidence-20261024` at `D:\Documents\Codex\VERDE_CONDITION3_EVIDENCE_20261024`.
- Verified this PC: Intel i3-9100, 16 GB RAM, Intel UHD 630, PyTorch 2.14.1+cpu, CUDA unavailable. Runtime package details are saved separately. This machine is not the collaborator's GTX 1650 device.
- Saved rollback copies of both experiment checkpoints under `D:\Documents\Codex\VERDE_CONDITION3_EVIDENCE_20261024\runs\rollback_20261024` and verified their SHA-256 values. The checkpoint copies are local; no duplicate weights are added to Git.
- Preserved and rechecked **11,228 files**, covering the restored image/annotation sources, canonical annotation working set, canonical V2 archive if present, and the repository model files. All hashes match the before-state.
- Reproduced the existing three-class view: **1,167 images / 1,282 boxes**; Healthy 233, Yellowing 549, Wilting 385. Train 760, validation 207, test 200. Copies use their own labels/cache paths; source labels/images remain unchanged.
- Verified 98 declared groups, zero declared-group split overlaps and zero exact binary overlaps across the historical splits. **Six source families span splits.** One inferred D1 filename/date sequence family spans train and test; this is a review flag, not a confirmed identity claim. Group metadata alone does not prove source/plant independence.
- Recorded the five already-known D1 historical JPEG/canonical hash differences. Baseline retesting uses the historical binaries and labels, with their actual hashes. No canonical transfers were performed in this task.
- Visually inspected an 18-image stratified historical sample and saved review flags without changing labels. It includes leaf-dominated crops, chlorosis-versus-wilt ambiguity and growing-context concerns. This is Codex review only.
- Downloaded **26 original files** from the CC BY 4.0 Fevgas water-stress dataset; 26/26 match publisher SHA-256. Extracted 26 embedded RGB JPEG streams losslessly and recorded byte linkage. The entire source has **two plants/groups**, not 26 independent plants. All remain **HOLD_GROWING_PLANT_CONTEXT**; zero are added to training. Source treatment was not assigned as a visual condition label. No pilot inference or annotation boxes were generated.

## Measured internal results

Both checkpoints were evaluated once on the same 200-image historical test split, 217 boxes, image size 416, CPU, batch 4, AP confidence floor 0.001, NMS IoU 0.7 and max detections 300. The evaluation commands and log are preserved. The installed runtime differs from the repository's pinned Ultralytics 8.4.174, but reproduced the recorded primary aggregate/AP results.

| Checkpoint | Class | Precision conf=.25 | Recall conf=.25 | F1 conf=.25 | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|---:|
| best_validation | Healthy | 79.03% | 94.23% | 85.96% | 86.87% | 65.05% |
| best_validation | Yellowing | 98.75% | 97.53% | 98.14% | 99.32% | 97.50% |
| best_validation | Wilting | 97.67% | 100.00% | 98.82% | 98.80% | 47.46% |
| final_gpu_resume | Healthy | 68.57% | 92.31% | 78.69% | 91.93% | 69.57% |
| final_gpu_resume | Yellowing | 98.75% | 97.53% | 98.14% | 98.37% | 96.95% |
| final_gpu_resume | Wilting | 97.67% | 100.00% | 98.82% | 98.33% | 47.32% |

Aggregate validation-selected: mAP50 **95.00%**, mAP50:95 **70.00%**. Aggregate final GPU-resumed: **96.21%**, **71.28%**.

Fixed-threshold precision/recall/F1 above use class-aware greedy IoU>=0.5 matching at conf>=0.25 against unchanged historical labels. AP uses the confidence sweep. Ultralytics' own P/R/F1 from its best mean-F1 point on this test curve are separately preserved and are descriptive, not a threshold selected on validation for deployment.

### Observed failures and benchmark limitations

- `D1-V006_77_1_10_05_03_12_1_2761e_20201209_47.jpeg` is historically Yellowing but validation-selected predicts Healthy at approximately 0.91. It is a leaf-dominated crop, so the example is not a complete growing-plant acceptance case.
- `B2-AGRIBOT-2025-04-30-126` has a historical box covering two distinguishable plants. Predictions separating those plants can score as extra boxes. Fixed-threshold FP/FN are disagreements with historical labels, not all confirmed biological failures. This label/instance issue must be reviewed before interpreting plant-separation metrics as final quality evidence.
- Final GPU-resumed Healthy fixed-threshold precision is lower despite higher aggregate AP50. That is another reason not to select a model solely by aggregate AP.
- Wilting AP50:95 remains approximately 47% for both checkpoints. Example boxes can include pot/soil or shifted foliage boundaries despite passing IoU=.5.
- Four earlier internet challenges and the five-photo PC check remain development evidence. Their original images and the six-epoch candidate are not in this fetched checkout. Their reported failures were not erased or relabeled as new measurements.

Annotated comparisons, class curves and confusion matrices are under `D:\Documents\Codex\VERDE_CONDITION3_EVIDENCE_20261024\experiments\condition3_evidence_20261024\evidence\error_review`. Machine-readable per-image errors, confidence values and hashes are included.

## New training and independent-test status

**New training approvals: 0. New training epochs: 0. Qualified untouched three-class acceptance images: 0. Independent per-class metrics: unavailable.**

The new source is held because planted/rooted context is not established, most early treatment frames do not establish visual wilting, and it contains no established Yellowing examples. Other leads are documented with access/rights limits. The repository describing 1,086 nutrient-deficiency originals does not distribute its dataset; its Apache code license is not evidence of image rights. No broad repeat search or unproductive unchanged-data fine-tune was used to manufacture a new checkpoint.

The >=85% mAP50 thesis target is a target. Existing internal scores exceed 85% on their historical benchmark but **do not establish independent acceptance or deployment readiness**.

## Rollback and exact continuation

- Retained validation-selected checkpoint: `D:\Documents\Codex\VERDE_CONDITION3_EVIDENCE_20261024\models\verde_condition3_best_validation.pt`.
- SHA-256: `54e4d45bde7eeb960d7458e28c8d535ce68745178cad668c47ba5086101fe987`.
- Final GPU-resumed comparison SHA-256: `d98118878dfb3b6fdd0bb5b71237f8bbdbbeb812c4acfc6b338e1a31cb049bf5`.
- No new best.pt/last.pt or training command exists for this task because training did not start.
- Evaluation command: `python experiments/condition3_evidence_20261024/evaluate_preserved_models.py`.
- New-data source/rights/assignments: `evidence/new_source_pilot/CODEX_VISUAL_REVIEW.csv` and `EMBEDDED_RGB_LINKAGE.csv`.
- Acceptance and live Pi requirements: `evidence/ACCEPTANCE_AND_PI_GATE.md`.
- Local savepoint: `D:\Documents\Codex\VERDE_CONDITION3_EVIDENCE_20261024\experiments\condition3_evidence_20261024\evidence\SAVEPOINT.json`.
- Isolated branch: `experiment/condition3-evidence-20261024`. The final response/ignored local push receipt records the evidence commit after push.

Continue by obtaining clearly documented growing-plant symptom data and an all-three-class source/plant-disjoint acceptance set, reviewing ambiguous historical instance labels in a separate corrected dataset, and running the next approved-data candidate on verified GPU hardware. Live Pi Camera Module 3 capture, timing/memory and rollback tests are still required. No Pi deployment package was prepared because independent acceptance did not pass. Unknown remains abstention; Harvest-Ready is a separate manual/date observation; Pest is a separate unvalidated task.

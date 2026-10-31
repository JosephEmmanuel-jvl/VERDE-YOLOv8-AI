# EXPERIMENTAL_CONDITION3_V1_CANDIDATE: evaluation report

**EXPERIMENTAL ONLY · INDEPENDENT_ACCEPTANCE_NOT_READY · DO NOT DEPLOY**

**Decision: `CANDIDATE_MIXED`.** Better box quality on the historical test (mainly Wilting) and better Healthy AP, but more Healthy false positives, no Yellowing gain, no overall gain on the validation split, and no real improvement on the four post-V1 images.

## Read this first: image size
The baseline was trained at imgsz 416 (12 epochs). The frozen template trains at 640. At 416 the baseline reproduces the historical record almost exactly (mAP50 95.00, mAP50-95 69.99; Wilting 47.43 vs 47.46 on record). At 640 it degrades sharply (Wilting recall@.25 48.81%), so the 640-vs-640 deltas requested as "same settings" overstate the candidate's gain. Three comparisons are therefore reported in `METRIC_TABLES.md`. The 416 runs were an added control declared after seeing the 640 results; they selected nothing.

| Overall, test | mAP50 delta | mAP50-95 delta | F1@.25 delta | Comment |
|---|--:|--:|--:|---|
| both at 640 | +2.59 | +13.67 | +9.77 | inflated: baseline off its trained size |
| both at 416 | +1.87 | +8.72 | -1.02 | fair same-size control |
| native (416 vs 640) | +2.41 | +6.96 | -3.69 | each model at its trained size |

Validation split (`validation_split_metrics.json`), overall candidate minus baseline: 640: AP50 -1.50, AP50-95 +1.33. 416: AP50 -0.44, AP50-95 +0.69. Native: AP50 -0.74, AP50-95 +0.01. Yellowing AP50 is lower in all three (-1.32, -1.92, -2.17). The large test Wilting AP50-95 gain does not appear on validation (native 90.46 to 93.77), so treat it as partly specific to this test set.

## Failure analysis (conf>=0.25, test split; annotated in `eval/<run>/failures/`, sheets in `eval/review_sheets/`)
Counts: baseline@640 | baseline@416 | candidate@640 | candidate@416
- TP of 217 boxes: 167 | 211 | 214 | 214. Images with any event: 94 | 10 | 15 | 9.
- Wilting missed: 43 | 0 | 0 | 0. Low-confidence Wilting: 35 | 0 | 0 | 0 (both only at baseline@640).
- Healthy background FP: 7 | 3 | 13 | 4. Healthy poor localization: 6 | 4 | 8 | 11. Healthy duplicates: 1 | 0 | 5 | 1. Merged plants: 0 | 1 | 2 | 5. Healthy missed: 6 | 4 | 1 | 1.
- Healthy as Yellowing: 1 | 1 | 0 | 1. Healthy as Wilting: 0 | 0 | 1 | 0. Yellowing as Healthy: 1 | 2 | 2 | 1. Yellowing as Wilting: 0 | 1 | 1 | 1. Wilting as Healthy: 0 in all. Wilting duplicates: 0 | 0 | 2 | 0.

Visual review (Claude, not human-reviewed): candidate duplicate boxes on single plants are genuine (COLL-HEALTHY-0035/0037, D1-compiled, RAW-WILTING-0369/0370). Several AgriBot "background FPs" are real unlabeled neighbouring plants, a label limitation that also affects the baseline. `D1-V006...2761e` is a leaf-dominated Yellowing crop both models call Healthy (already on record). `D1-rathor__N-2` is a drooping Yellowing plant called Wilting. `B4-WM-0129` is a pale Healthy plant that also gets a Wilting box.

## TRAINING-DATA SANITY CHECK: NOT GENERALIZATION EVIDENCE (candidate@640)
All 6 LettuceDeF found as Yellowing (conf 0.91-0.95; LETTUCEDEF_006 has an extra box, min 0.64). All 36 LAB boxes found as Wilting, no wrong class, but 52 predictions against 36 boxes (min conf 0.27): extra boxes on single leaves and neighbouring fragments, plus low-confidence Wilting on plumbing hardware in Cam1/Cam17. Some fragments may be the 3 intentionally unboxed regions. For contrast, the baseline found 0/12 at 640 and 1/12 records at 416, calling these plants Healthy.

## POST-V1 DEVELOPMENT CHALLENGE: NOT INDEPENDENT ACCEPTANCE
No tuning used these images. Yellowing detected at conf>=0.25, IoU>=0.5 with the approved box:

| Image | baseline@416 | candidate@640 | baseline@640 | candidate@416 |
|---|---|---|---|---|
| g13-bg21 | Healthy 0.70 | no detection >=0.05 | Healthy 0.25 | Healthy 0.37 |
| g13-bg19 | no detection | Healthy 0.42 + 0.22 | Healthy 0.24 | Healthy 0.07 |
| USDA N img05 | Healthy 0.91 | Healthy 0.90 (box stops above the yellow leaves, IoU 0.70) | Healthy 0.82 | Healthy 0.94 |
| USDA P img05 | Healthy 0.90 | Healthy 0.92 | Healthy 0.85 | Yellowing 0.48 with Healthy 0.58 and Wilting 0.19 on the same plant |
| Yellowing correct | 0/4 | 0/4 | 0/4 | 1/4 (weak, Healthy scores higher) |

At each model's trained size the candidate is 0/4, the same as the baseline, and it did not generalize Yellowing to these sources. Four images from two families cannot prove anything alone, but the result agrees with the flat Yellowing metrics.

## Limits
- **One arm only:** the handoff planned matched historical-control vs plus-new arms. The candidate starts from the baseline and also changes image size and training length, so differences cannot be attributed to the 12 new records.
- **Historical test is not independent:** six source families span historical splits (on record), and this test set has now been read at two sizes. It must not be used for further selection.
- **Partial-scene risk accepted:** `--acknowledge-partial-scene-risk` was passed, per the authorized 1,179-image dataset. Three unboxed LAB regions may train as background.
- **Small test classes:** Healthy has 52 boxes, so small deltas are noise.
- **Fixed P/R differs from the record:** baseline@416 P/R@.25 differ somewhat from the older table (Healthy P 83.93 vs 79.03) although AP matches. The earlier run was CPU on a different machine and library set; not investigated further.

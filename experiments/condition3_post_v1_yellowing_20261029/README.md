# POST-CONDITION3-V1 YELLOWING EVIDENCE

**`TRAIN_CANDIDATE`** · **`NOT PART OF FROZEN CONDITION3-V1`** · **`NOT INDEPENDENT ACCEPTANCE EVIDENCE`** · **`CONDITION3-V2 NOT YET BUILT`** · **`DO NOT DEPLOY`**

Four human-reviewed Yellowing images found after the frozen Condition3-V1 experiment
(`experiment/condition3-new-symptom-training-20261024` @ `2f2849c7de94df0b7a416c14eca6b9c40456332e`).
They are intended for a future, separately versioned **Condition3-V2, which does not exist yet**. Nothing here has been trained on, merged into a dataset, or added to V1.
Every record has the role `POST_CONDITION3_V1_TRAIN_CANDIDATE`. Independent acceptance status: `INDEPENDENT_ACCEPTANCE_NOT_READY`.

| | Images | Boxes | Physical-plant/group records | Source-family blocks |
|---|---|---|---|---|
| Coon / University of Florida (`COON_UF_HEAT_STRESS_STUDY`) | 2 | 2 | 2 (`COON_UF_G13`, `COON_UF_G13_BG19`) | 1 |
| USDA-ARS (`USDA_ARS_ATRU_FRANTZ_HYDROPONIC_BUTTERHEAD`) | 2 | 2 | 2 (`ARS_ATRU_N_MATURE`, `ARS_ATRU_P_ADVANCED`) | 1 |
| **Total** | **4** | **4** | **4** | **2** |

## Image count is not source independence
Four images do not mean four independent sources. Both Coon plants come from one chamber run and one capture session (2025-06-25). Both USDA photos come from one research unit's fact-sheet series, with no published plant IDs or capture dates. Keep each source-family block on one side of any split, and do not use these images as acceptance evidence.

## Contents
- `images/yellowing/`: original image files, unmodified
- `labels/yellowing/`: YOLO labels, class `1` = Yellowing, one box per image
- `manifests/`: annotations, provenance and licences, leakage groups, source families, post-V1 manifest
- `records/human_review_decisions.json`: human class and box decisions for these four images
- `rights/`: USDA rights verification and the Zenodo record for the Coon data
- `reports/`: final Yellowing report (written before this package; its local paths refer to the original workspace) and `label_round_trip.png`
- `ATTRIBUTION.md`, `RECOVERY.md`, `SHA256SUMS.txt`, `verify_package.py`

## Label notes
Labels match the stored pixels of the committed files. The Coon images have EXIF orientation 1 (no rotation) and the USDA images have no EXIF orientation tag.
`Butterhead_Phosphorus_img05.jpg` has a documented ambiguity: a top-edge leaf `[945, 0, 1095, 345]` that may belong to a neighbouring plant stays inside the approved box, and a right-edge leaf `[1095, 250, 1176, 700]` is excluded. Both USDA plants run out of frame; see `clipping_ambiguity` in `manifests/annotations.json`.

## Verify
```bash
python verify_package.py
```

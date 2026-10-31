# VERDE — final post-V1 Yellowing evidence (2026-10-29)

Status flags: **`PUBLIC_YELLOWING_SEARCH_EXHAUSTED_FOR_NOW`** · **`WILTING_WORK_PAUSED_HERE`** · frozen V1 `2f2849c7de94df0b7a416c14eca6b9c40456332e` **untouched** · not merged, not trained.

| | Coon (UF heat stress) | USDA-ARS (ATRU hydroponic butterhead) | **Total post-V1** |
|---|---|---|---|
| Approved images | 2: g13-bg21, g13-bg19 (top) | 2: N img05, P img05 | **4** |
| Approved boxes | 2 | 2: `[0,0,615,745]`, `[15,0,1095,1139]` | **4** |
| Physical-plant groups | 2: `COON_UF_G13`, `COON_UF_G13_BG19` | 2: `ARS_ATRU_N_MATURE`, `ARS_ATRU_P_ADVANCED` | **4** |
| Source-family blocks | `COON_UF_HEAT_STRESS_STUDY` (one session 2025-06-25) | `USDA_ARS_ATRU_FRANTZ_HYDROPONIC_BUTTERHEAD` | **2** |
| Rights | CC BY 4.0 (Zenodo 17458168) | PUBLIC_DOMAIN_VERIFIED_NO_THIRD_PARTY_NOTICE | |
| Role | Training candidate only | Training candidate only | Acceptance evidence: **0** |

Held, not admitted: USDA N img03 and P img06 (`HUMAN_REVIEW_HOLD`).
P img05 keeps its documented ambiguity: the top-edge leaf `[945,0,1095,345]` (4.2% of box) is retained, the right-edge leaf `[1095,250,1176,700]` is excluded, and the plant is clipped at top and bottom. The tighter x≈945 box was rejected by the human reviewer. N img05: clipped at left and top, right-edge neighbours excluded.
Coon boxes are in the EXIF-oriented frame; apply `exif_transpose` before using their labels.

Frozen V1 (`experiment/condition3-new-symptom-training-20261024`): 1,179 images, 555 of them Yellowing (549 historical + 6 LettuceDeF). None of the 4 post-V1 images is in V1 by SHA-256. They belong only to a future V2 experiment, with each family block kept on one side of any split.

## Files
- Final annotations: `annotations/FINAL_POST_V1_YELLOWING_ANNOTATIONS.json`; YOLO labels in `annotations/yolo_final/` (class 1 = Yellowing); visual check `POST_V1_YELLOWING_FINAL_BOXES.png`
- Human-review ledgers: `metadata/human_review_USDA_20261029.json`; Coon: `..\VERDE_YELLOWING_NEW_SOURCE_ACQUISITION_20261029\metadata\human_review_20261029.json`
- Provenance/licence: `metadata/image_ledger.json`, `metadata/USDA_RIGHTS_VERIFICATION.json`
- Source matrix: `metadata/FINAL_YELLOWING_SOURCE_MATRIX.json`
- Post-V1 manifest: `metadata/POST_V1_YELLOWING_EVIDENCE_MANIFEST.json`
- Unsent author requests: `AUTHOR_REQUESTS_UNSENT.md`
- Checksums: `SHA256SUMS.txt` (this workspace) and the Coon workspace's `SHA256SUMS.txt`
- Savepoint: `savepoints/RESUME_SAVEPOINT_FINAL_20261029.txt`

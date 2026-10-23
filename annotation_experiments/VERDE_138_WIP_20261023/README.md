# VERDE 138-Image Assisted Annotation Experiment

> **WORK IN PROGRESS — NOT CANONICAL — NOT TRAINING READY**

This directory is an isolated experimental snapshot of Codex-generated/assisted annotations for the remaining VERDE plant-condition review queue. It is preserved for collaboration and resumable QA only.

## Snapshot state

- Images processed: 138/138
- Candidate YOLO annotations: 125 images
- Candidate AnyLabeling JSON: 138 records
- Unannotated/held: 13 images
- Proposed boxes: 800
- Classes: Healthy 36, Yellowing 1, Harvest-Ready 101
- Final visual QA: incomplete
- Final integrity/checksum validation: incomplete

No image binaries are duplicated in this Git snapshot. Records are linked to canonical images by SHA-256 and source paths in the annotation ledger.

## Critical restrictions

- These labels must not replace verified human annotations.
- These labels must not be merged into canonical V2 without later authorization and completed QA.
- These labels must not be used as the final training dataset.
- Current QA fields are preliminary and are not supervisor approval.

## Known QA issue

A visual spot-check after this build found that some records currently labeled `QA_PASS`, particularly dense UC/Aalto layouts, may retain grid-like/template alignment. A conservative QA rebuild was started but did not complete because the temporary image-processing environment was invoked with an incompatible Python interpreter. Treat every current pass as pending re-review.

## Contents

- `candidate_yolo_labels/` — candidate YOLO label files for 125 images
- `candidate_anylabeling_json/` — candidate AnyLabeling records for all 138 images, including held/unannotated records
- `VERDE_HUMAN_STYLE_ANNOTATION_LEDGER_20261023.csv` — SHA-linked candidate inventory
- `VERDE_HUMAN_STYLE_QA_LEDGER_20261023.csv` — preliminary QA data
- `VERDE_HUMAN_REFERENCE_COMPARISON_LEDGER_20261023.csv` — human-style comparison status
- `VERDE_HUMAN_REFERENCE_SAMPLE_INDEX_20261023.csv` — reference sample index
- `VERDE_HUMAN_STYLE_ASSISTED_REANNOTATION_SAVEPOINT_20261023.json` — machine-readable build counts
- `WIP_PAUSE_SAVEPOINT_20261023.md` — exact pause/resume instructions and limitations

## Resume requirement

Resume from the external experimental workspace recorded in `WIP_PAUSE_SAVEPOINT_20261023.md`. Complete conservative visual QA, regenerate affected ledgers/previews, and perform checksum/integrity validation before proposing any canonical merge.


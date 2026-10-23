# VERDE Human-Style Assisted Reannotation — WIP Pause Savepoint

**WORK IN PROGRESS — NOT CANONICAL — NOT TRAINING READY**

Paused on 2026-10-23 before final QA and integrity reporting.

## Current experimental state

- Images processed: 138/138
- Images with candidate YOLO labels: 125
- Unannotated/held images: 13
- Proposed plant-instance boxes: 800
- Candidate AnyLabeling JSON records: 138
- Canonical classes represented: Healthy 36, Yellowing 1, Harvest-Ready 101
- Training started: no
- Canonical V2 modified: no
- Verified human annotations modified: no

## Important QA limitation

The latest completed workspace still contains preliminary QA labels from the automated build. A subsequent visual spot-check found that records currently marked `QA_PASS`, especially dense UC/Aalto layouts, may retain grid-like/template alignment. A conservative QA rebuild was started but did not complete because the temporary image-processing environment was invoked with an incompatible Python interpreter.

Therefore:

- no current `QA_PASS` value should be treated as supervisor approval;
- final visual QA remains incomplete;
- final checksum/integrity validation remains incomplete;
- candidate annotations must not replace verified human annotations;
- candidate annotations must not be used as a final training dataset.

## Resume point

Resume from this workspace. Re-run the conservative QA build with a Python 3.12-compatible image stack, visually inspect all proposed pass/adjust groups, regenerate ledgers and contact sheets as needed, then create and verify the final checksum manifest and report.


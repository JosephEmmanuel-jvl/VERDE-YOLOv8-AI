# VERDE condition experiment evidence — October 24, 2026

EXPERIMENTAL — DATA REVIEW AND PRESERVED-MODEL RETEST — DEPLOYMENT GATE NOT PASSED

This isolated branch continues the corrected October 24 handoff. It records actual local data review and CPU evaluation. No new training was started because no new symptom training records passed the growing-plant/context gate, no qualified independent three-class acceptance set was available, and this PC has no CUDA device. Repeating the unchanged-data fine-tune would not resolve the handoff's reported generalization failures.

The existing validation-selected YOLOv8s checkpoint remains the experimental rollback candidate. No production/Pi model has been inspected, changed, or verified here.

## Reproduction

Run the scripts from the repository root with Python 3.12 and the recorded runtime packages. `evidence/RUNTIME.json` and `evidence/INSTALLED_DISTRIBUTIONS.json` record the runtime actually used. The repository originally pins Ultralytics 8.4.174; this CPU retest used an existing 8.4.184 installation. The original and retest measurements are kept separately.

1. `python experiments/condition3_evidence_20261024/audit_and_restore.py`
2. `python experiments/condition3_evidence_20261024/intake_stress_pilot.py`
3. `python experiments/condition3_evidence_20261024/extract_embedded_rgb.py`
4. `python experiments/condition3_evidence_20261024/record_visual_findings.py`
5. `python experiments/condition3_evidence_20261024/evaluate_preserved_models.py`
6. `python experiments/condition3_evidence_20261024/error_review.py`

The intake script resumes a partial build by checking copied hashes. It refuses to overwrite a completed audit. The evaluator retains existing completed reports and uses separate result directories. `dataset/` and `runs/` remain ignored; exact source/image/label hashes and paths are recorded in evidence manifests.

The model class map is `0 Healthy`, `1 Yellowing`, `2 Wilting`. Unknown is an abstention, not class 3. Harvest status is a separate manual/date-based observation in this experiment. Pest is outside this condition model's validated scope. The 138-image WIP was not used for symptom training or evaluation.

## New-source attribution and status

Fevgas, Georgios; Lagkas, Thomas; Argyriou, Vasileios; Sarigiannidis, Panagiotis (2025), “A Dataset of Thermal Infrared (TIR) and RGB Images of Lettuces, soil moisture data, and Pseudo-coloring RGB Images of the Lettuce's Stressed Areas,” Mendeley Data, V2, DOI: [10.17632/294zk6k5wf.2](https://data.mendeley.com/datasets/294zk6k5wf/2). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Twenty-six original files were downloaded and matched the publisher's SHA-256 values. Embedded visible-spectrum JPEG streams were extracted without pixel modification; byte offsets, original/child hashes, owner and URLs are in `evidence/new_source_pilot/EMBEDDED_RGB_LINKAGE.csv`. The original thermal display is not an RGB condition-training input. Contact sheets resize the embedded RGB images only for review.

All 26 are held in source quarantine. They depict two repeated plants/heads on a tabletop; growing/rooted context is not established by the visible image. The source's water-stress treatment is not converted into a Wilting label. No Yellowing examples were established. No boxes were generated for these held records. They have not been passed through either model and must not be called an evaluated or qualified final acceptance set.

## Review limitations

Codex reviewed the new 26-image source pilot and an 18-image stratified historical annotation sample. These are Codex observations, not human-supervisor approval or a full visual approval of all 1,167 historical images. Historical labels are unchanged.

Some historical benchmark errors require annotation interpretation. In particular, `B2-AGRIBOT-2025-04-30-126` contains a historical box spanning two distinguishable plants. A prediction separating those plants can score as a false positive against that box. Fixed-threshold FP/FN tables measure disagreement with historical labels; they are not all proven botanical errors. The separate visual-review ledger preserves this concern.

`evidence/FINAL_STATUS.md` contains the measured results, blockers and exact continuation instructions. `evidence/ACCEPTANCE_AND_PI_GATE.md` records the gates still required before any deployment package is justified.

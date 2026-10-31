# Recovery and savepoint

State: run complete and archived. `CANDIDATE_MIXED`, `EXPERIMENTAL_ONLY`, `INDEPENDENT_ACCEPTANCE_NOT_READY`, `CONDITION3_V2_NOT_YET_BUILT`, `DO_NOT_DEPLOY`. No training is queued.

- Branch base: `main` @ `9c410c435d2ed5e883da3e64b92167fd74ee7593`. Not merged into `main`.
- Source experiment, unchanged: `experiment/condition3-new-symptom-training-20261024` @ `2f2849c7de94df0b7a416c14eca6b9c40456332e`.
- `weights/best.pt` sha256 `e6b3182c9edc67e51a65d660c0566122b85f2d0a9d173f6a133c81d55d40ffe9`; `weights/last.pt` sha256 `1b9142fd118e57f3bff6e77ecc418d5b6a73ceadad3fe5ac7ce170a2e4fba5d8`. Both are committed here.
- Original run workspace (not in Git): `C:\Users\ben\Documents\Codex\VERDE_CONDITION3_V1_TRAINING_20261031`. It also holds the 1,179-image dataset build, per-epoch checkpoints and full-size logs.
- To rebuild the dataset: run `scripts/build_experiment.py --variant plus-new --acknowledge-partial-scene-risk` from the V1 handoff package (see `SOURCE.json`), then compare with `dataset_manifest/BUILD_RECEIPT.json`.
- Do not re-run training for this candidate, and do not use the historical test set for further model selection.
- Original savepoint: `savepoints/RESUME_SAVEPOINT.txt`.

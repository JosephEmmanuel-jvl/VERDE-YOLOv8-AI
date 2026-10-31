# Recovery and savepoint

State at publication: `CONDITION3_V2_NOT_YET_BUILT`, `INDEPENDENT_ACCEPTANCE_NOT_READY`, `DO_NOT_DEPLOY`, `PUBLIC_YELLOWING_SEARCH_EXHAUSTED_FOR_NOW`. Coon acquisition is closed. No training has used these images.

- Branch base: `main` @ `9c410c435d2ed5e883da3e64b92167fd74ee7593`. This branch has not been merged into `main`.
- Frozen V1, unchanged: `experiment/condition3-new-symptom-training-20261024` @ `2f2849c7de94df0b7a416c14eca6b9c40456332e`.
- Intermediate acquisition material stays outside Git, in the local workspaces `VERDE_YELLOWING_NEW_FAMILIES_20261029` and `VERDE_YELLOWING_NEW_SOURCE_ACQUISITION_20261029`. Source hashes are in `manifests/post_v1_manifest.json`.
- If those workspaces are lost, re-fetch the originals from the URLs in `manifests/provenance_license.json` and compare with the SHA-256 values there.
- Excluded on purpose: USDA N img03 and P img06 (human hold), rejected Coon G14 plants, other Coon frames, and the g13-bg19 side view.

Next step needing a human decision: whether and when to build Condition3-V2.

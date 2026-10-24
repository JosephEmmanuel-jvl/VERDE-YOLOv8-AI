# Acceptance and Raspberry Pi gate — pending

No new training or acceptance inference has occurred. Qualified independent acceptance images: **0**. All new-source pilot records remain quarantined. Existing four internet challenges and the collaborator's five-photo PC check are development evidence, not untouched holdout images; the raw nine files are not available in this checkout and their hashes/labels must not be invented. The six-epoch augmentation checkpoint reported in the handoff is also absent from the fetched repository, so it was not re-evaluated here.

## Before the next training run

1. Obtain whole growing lettuce photographs showing clear Healthy, genuine chlorosis/Yellowing, and structural Wilting. Keep rot, leaf-only crops, harvested produce and ambiguous symptoms in separate review queues.
2. Record original SHA-256, original filename, source URL/owner, exact permission/license, capture date if known, plant ID, scene/session ID, reviewer and visual rationale. Unknown dates stay unknown.
3. Apply one box per distinguishable plant, outer visible foliage, excluding unnecessary pot/soil/equipment. Visually review all new labels. Mixed or uncertain conditions require an explicit human decision rather than forced relabeling.
4. Reserve acceptance groups before training. Use new plants/captures and a source family absent from training and model-selection validation. Keep every plant, session, derivative and sequence family in one partition. Freeze image/label hashes and assignments; never use acceptance images for augmentation, early stopping, threshold selection or model selection.
5. Include all three classes and realistic multi-plant scenes, lighting/background variation and hard negatives. No particular sample size or new pass threshold has been authorized beyond the thesis's >=85% mAP@0.5 target; agree a defensible class/plant count and class-specific recall/FP criteria before model selection. A tiny two-plant laboratory sequence cannot establish field acceptance.
6. Resolve the filename-derived potential D1 sequence overlap and review incomplete/merged historical plant boxes in an isolated corrected dataset, preserving the historical benchmark for comparison.
7. Use a verified GPU runtime for an isolated candidate only after these data gates pass. Record the exact approved training manifest and data hash, seed, config, package versions, command, epoch logs and best/last hashes. No command in this evidence branch automatically starts training.

## Independent evaluation and Unknown

Report per-class precision, recall, F1, AP50 and AP50:95 with image/plant counts. Preserve annotated misses, wrong classes, merged/split instances and confidence values. Use the same fixed image size, NMS/IoU and confidence policy for baseline/candidate comparisons. AP sweeps confidence; operating-point precision/recall must be reported separately. Unknown means no supported condition output passed the locked threshold; it is not proof of an unhealthy or healthy state.

## Live Pi Camera Module 3 evidence still needed

Only after independent acceptance passes should an isolated Raspberry Pi 5 package be prepared. Before modifying production, separately authorize and verify on-device model format and dependencies, class map, model checksum, image size and normalization, camera orientation/colour pipeline, threshold/Unknown behavior, capture-to-result latency, FPS, CPU temperature/throttling, peak memory and repeated-run stability. Test the actual Camera Module 3, multi-plant separation and challenging illumination. Preserve and hash the deployed weights/config for rollback and test rollback locally. PC CPU timing is not Pi timing. Backend, camera service, app, automation, dosing, SMS and production weights remain outside this task.

# Middle-six condition retraining — 28 October 2026

## Saved candidate

- Three-class YOLOv8 condition weights: `weights/best.pt` (best epoch 3 of 6 run epochs; early stopping after three non-improving epochs).
- SHA-256: `36C4155AF38AF0E9EAAC4CF8C487B24D6A6790F57C3F144A71C69908B0F0269C`.
- Started from the preserved mounted-condition V1 checkpoint. The Pi worker and deployed weights were not changed.
- Training used an NVIDIA GTX 1650 Max-Q; maximum observed GPU temperature was 87°C.

## Image use

- 1,336 training crops: all 860 inherited reviewed train crops, plus 476 newly prepared crops. The 207 inherited validation images were retained.
- New crops used 130 of 192 aligned phone photos, all 12 aligned 27 October phone photos, and five reviewed plant crops from the 28 October center-six photo. All eligible repeated views were kept, including duplicate originals. No file was removed for similarity.
- The earlier Pi batch contributes reviewed crops through the inherited set. `IMAGE_USE_AUDIT_20261028.md` and the provenance CSVs document other original photos that were examined or used in separate experiments.
- **Not every supplied photo entered this supervised run.** Sixty-two of the 192 phone images lacked accepted plant alignment, and other Pi/earlier originals lack complete reviewed boxes. Assigning guessed boxes or empty labels would train the model on false targets. The original photos remain saved for annotation.
- New crops added 203 Healthy and 273 Wilting examples; Yellowing was present only in the inherited dataset. This run does not train Harvest-Ready or printed pest detection.

## Checks

| Check | Mounted V1 | New candidate | Limit |
| --- | ---: | ---: | --- |
| Reviewed repeated Pi crops, Healthy top class | 16/16 | 16/16 | Same known plants and setup |
| Reviewed repeated Pi crops, Wilting top class | 31/32 | 32/32 | Same known plants and setup |
| Reviewed repeated Pi crops, Wilting at confidence >= 0.50 | 21/32 | 28/32 | Threshold is provisional |
| Historical validation mAP50 (all three classes) | — | 0.925 | Not independent field accuracy |

The 48 crop comparison cases were not added to this training dataset, but were inspected in earlier development. See `field_comparison.json` for every individual prediction.

On a saved 4608×2592 Pi camera frame, the full-frame localizer found eight plant regions, but missed a visibly severe wilt near the top center. An isolated crop of the center six, using 960-pixel localizer input, found only three regions and still missed the wilted plants (`pi_center_six_from_field/candidate-annotated.jpg`). Changing model input size alone also produced duplicate localization on another check. The condition model cannot classify plants the localizer does not find. A separate phone photo of the same center-six scene produced better boxes, but five crops from that exact photo were included in training; this is not an independent accuracy check.

## Status

This is a **saved local condition candidate**, not a verified replacement for the Pi's five-class worker. Next, annotate missing high-angle originals and improve/test the localizer on independent Pi frames. Then compare condition, Harvest-Ready, and printed-prop results together on one fresh camera capture before changing the live worker or app.

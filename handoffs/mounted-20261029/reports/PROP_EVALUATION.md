# VERDE local Wilting and printed-prop check — 28 October 2026

This is a PC-only experiment. No Raspberry Pi weight, service, backend, camera
setting, or application file was changed.

## Wilting and Yellowing

The preserved three-class condition V1 and the saved middle-six V2 were checked
on the same inherited 207-image validation split. Per-class mAP50:

| Class | V1 | V2 |
| --- | ---: | ---: |
| Healthy | 0.886 | 0.890 |
| Yellowing | 0.925 | 0.912 |
| Wilting | 0.976 | 0.954 |

On 48 repeated reviewed Pi-camera crops of the same plants, V2 classified
28/32 Wilting crops correctly at a 0.50 confidence gate, versus 21/32 for V1.
These are development comparisons, not independent accuracy estimates. V2
helps this particular Wilting plant while the inherited validation Wilting and
Yellowing mAP50 values fell slightly. There is no mounted-setup yellowing plant
to verify Yellowing performance. More epochs on the same repeated images are
unlikely to answer that question.

## Printed insect props

`printed_prop_fixture_v7.pt` continues the preserved raised-camera v6 model
for four epochs with 32 reviewed no-on-leaf-prop tiles from Pi-camera sets
03–04. Sets 09–12 were kept out of this continuation training. One original
image per set was used for the comparison at confidence 0.25 with 640-pixel
tiles and box suppression. The four sets are the same physical setup and are
not an independent deployment sample.

| Model | On-leaf props matched, sets 09–10 | Extra boxes in those scenes | False boxes, clear sets 11–12 |
| --- | ---: | ---: | ---: |
| Preserved v6 | 7/7 | 4 | 3 |
| Experimental v7 | 7/7 | 2 | 2 |

Matched v7 scores in these two positive scenes were 0.773–0.909. Its four
extra/false scores were 0.508–0.612. A 0.70 gate would separate those particular
boxes, but it is not calibrated for new scenes; raising the gate could miss
smaller props. The saved 28 October `field-frame.jpg` is a *photo of a display*,
not an original Pi-camera JPEG. Both models found only one candidate at 0.25
(v6: 0.344; v7: 0.272). This does not establish either model can find all four
props in that view.

Keep v7 experimental. The preserved v6 and current Pi weights remain available
for rollback. Before changing the live app, compare both on original mounted
Pi-camera photos from a new placement and a genuinely clear scene, with reviewed
prop boxes and complete camera-to-app timing.

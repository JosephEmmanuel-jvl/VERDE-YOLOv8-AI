# VERDE AI training continuation

Training and evaluation files for the VERDE Healthy / Yellowing / Wilting
YOLOv8s experiment.

This is an earlier three-class experiment; the final thesis system uses four
plant states: Healthy, Yellowing, Wilting and Harvest-Ready.

## My role

I collected the dataset, annotated the images, and trained the YOLOv8 model.

## Limitations and next steps

This is an experimental three-class model. It scored `0.962 mAP50` and
`0.713 mAP50-95` on the existing 200-image VERDE V2 test split, which is not
yet proven independent by plant or source. The four-photo external development
challenge showed that performance did not generalize reliably.

The next steps are to build an independent plant/source/date-disjoint test
split and test on external data, including new Raspberry Pi camera images.
These checks will guide further training and are needed before treating this
candidate as a validated replacement for a deployed model.

Classes:

| ID | Class |
| ---: | --- |
| 0 | Healthy |
| 1 | Yellowing |
| 2 | Wilting |

## Repository contents

- `models/verde_condition3_best_validation.pt` - validation-selected starting
  checkpoint for the next isolated experiment.
- `models/verde_condition3_final_gpu_resume.pt` - final checkpoint from the
  recovered GTX 1650 run, retained for comparison.
- `reports/` - measured results and original training arguments.
- `src/prepare_dataset.py` - rebuilds the three-class dataset without changing
  the five-class source dataset.
- `src/train.py` - starts a new, configurable training run from a selected
  checkpoint.
- `src/evaluate.py` - evaluates a checkpoint on a named dataset split.
- `src/verify_setup.py` - verifies model hashes, class names, CUDA, and dataset
  structure before training.

The required source dataset is stored as split Git LFS archive parts under
`dataset_sources/`. Start with [`HANDOFF.md`](HANDOFF.md), then follow
[`DATASET_TRANSFER.md`](DATASET_TRANSFER.md) to verify, join, and extract the
dataset before preparing the isolated three-class dataset.

## Paths and retained artifacts

Historical report paths use `<EXPERIMENT_ROOT>` as a placeholder for the
original experiment workspace. Replace it with your local path when reusing
archival training YAMLs; it is not a path the training tools resolve automatically.
Reported metrics and training parameters have not been changed.

The tracked `dataset_sources/*.zip.part*` files are intentionally retained
as Git LFS dataset parts. Joined ZIPs and extracted local data are ignored.
Checkpoints and dataset archives are unchanged; embedded metadata inside
those binary artifacts may still contain historical local paths.

## New-device setup

Use Python 3.12. Create and activate a virtual environment, then install the
CUDA-enabled PyTorch build appropriate for the new GPU and driver from the
official PyTorch selector. After PyTorch reports `torch.cuda.is_available()` as
`True`, install this repository's packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Git LFS before cloning, then restore the two dataset folders:

```powershell
python src\restore_dataset_archives.py --dest "D:\VERDE_AI_DATA"
```

After extraction, run:

```powershell
python src\prepare_dataset.py `
  --source "D:\VERDE_AI_DATA\verde_v2" `
  --decoded "D:\VERDE_AI_DATA\verde_v2_anylabeling" `
  --dest "dataset\three_class"

python src\verify_setup.py --data "dataset\three_class\data.yaml"
```

Start a new isolated run. `--batch -1` lets Ultralytics estimate a batch size
for the stronger GPU:

```powershell
python src\train.py `
  --data "dataset\three_class\data.yaml" `
  --model "models\verde_condition3_best_validation.pt" `
  --epochs 50 --patience 10 --imgsz 640 --batch -1 --device 0 `
  --name "condition3_stronger_gpu_v1"
```

Evaluate without changing the test split:

```powershell
python src\evaluate.py `
  --model "runs\condition3_stronger_gpu_v1\weights\best.pt" `
  --data "dataset\three_class\data.yaml" `
  --split test --imgsz 640 --device 0
```

Do not tune repeatedly against the test split or the old four-photo challenge.
The next defensible acceptance gate is a plant/source/date-disjoint holdout of
whole growing lettuce, including Pi Camera Module 3 images.


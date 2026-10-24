# VERDE YOLOv8 stronger-GPU handoff

Date: 2026-10-23

## Mission

Continue the isolated VERDE three-class YOLOv8 experiment on a stronger NVIDIA
GPU. The goal is a model that reliably distinguishes Healthy, Yellowing, and
Wilting lettuce under real hydroponic and Raspberry Pi camera conditions. Do
not optimize only for the existing split; generalization is the priority.

## What has been completed

- Recovered and completed the local YOLOv8s training run on a GTX 1650.
- Preserved the validation-selected checkpoint and the final resumed-run
  checkpoint under `models/`.
- Preserved measured evaluation JSON, training CSV, and arguments under
  `reports/`.
- Added portable dataset preparation, setup verification, training, and
  evaluation scripts under `src/`.
- Packaged both required source-data folders into verified split Git LFS
  archives under `dataset_sources/`.

Measured results on the existing 200-image VERDE V2 test split:

| Checkpoint | mAP50 | mAP50-95 | Wilting AP50-95 |
| --- | ---: | ---: | ---: |
| Validation-selected | 0.950 | 0.700 | 0.475 |
| Final GPU-resumed | 0.962 | 0.713 | 0.473 |

These are real measured results, but the split is not proven independent by
plant, source, or capture date. The candidate also failed a four-photo external
development challenge. It is therefore experimental and not deployment-ready.

## Start here on the stronger device

1. Install Git and Git LFS, then clone this private repository.
2. Run `git lfs pull` and confirm all four archive parts are large binary files.
3. Install Python 3.12 and the CUDA-enabled PyTorch build matching the new GPU.
4. Create the environment and install `requirements.txt`.
5. Restore the source data:

   ```powershell
   python src\restore_dataset_archives.py --dest "D:\VERDE_AI_DATA"
   ```

6. Prepare and verify the isolated three-class dataset:

   ```powershell
   python src\prepare_dataset.py `
     --source "D:\VERDE_AI_DATA\verde_v2" `
     --decoded "D:\VERDE_AI_DATA\verde_v2_anylabeling" `
     --dest "dataset\three_class"

   python src\verify_setup.py --data "dataset\three_class\data.yaml"
   ```

   Expected counts: 760 train, 207 validation, and 200 test images.

7. Start a new isolated run from the validation-selected checkpoint:

   ```powershell
   python src\train.py `
     --data "dataset\three_class\data.yaml" `
     --model "models\verde_condition3_best_validation.pt" `
     --epochs 50 --patience 10 --imgsz 640 --batch -1 --device 0 `
     --name "condition3_stronger_gpu_v1"
   ```

8. Evaluate the best checkpoint once on the unchanged test split:

   ```powershell
   python src\evaluate.py `
     --model "runs\condition3_stronger_gpu_v1\weights\best.pt" `
     --data "dataset\three_class\data.yaml" `
     --split test --imgsz 640 --device 0
   ```

## What still needs to be done

- Confirm CUDA, GPU model, VRAM, PyTorch version, and Ultralytics version in the
  new run log.
- Train the stronger-GPU experiment without changing the existing test split.
- Save `best.pt`, `last.pt`, `results.csv`, training arguments, and evaluation
  JSON for reproducibility.
- Build a genuinely independent acceptance set grouped by whole plant, image
  source, and capture date. Include full growing lettuce captured by the Pi
  Camera Module 3 under varied lighting and backgrounds.
- Compare class-level recall and AP, especially Wilting, rather than accepting
  a model based only on aggregate mAP.
- Only after the independent camera test passes should deployment/export work
  for Raspberry Pi 5 begin.

## Safety and scientific-integrity rules

- Do not replace the current five-class VERDE production model with this
  three-class candidate.
- Do not claim deployment readiness from the recorded test metrics alone.
- Do not tune repeatedly against the 200-image test split or the old four-photo
  development challenge.
- Do not merge Harvest-Ready or Pest Detected labels into Healthy. The
  preparation script excludes images containing those labels.
- Keep all new runs in separate folders so previous checkpoints and evidence
  remain reproducible.

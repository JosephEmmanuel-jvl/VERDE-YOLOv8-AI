"""Isolated, thermally guarded training. No Pi, deployment, or dataset writes outside results."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import torch
from ultralytics import YOLO

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
WORK = OUT/'_work'
last_check = 0.0
TASK = ''

def thermal_guard(trainer):
    global last_check
    if time.monotonic()-last_check < 5:
        return
    cooldown_start = None
    while True:
        temperature = int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'], text=True).splitlines()[0].strip())
        with (OUT/f'{TASK}_thermal.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'temperature_c': temperature, 'epoch_zero_based': getattr(trainer, 'epoch', None)})+'\n')
        last_check = time.monotonic()
        if temperature >= 85:
            raise RuntimeError('THERMAL_SAFE_STOP: GPU reached 85 C; previous saved checkpoints preserved')
        if cooldown_start is None and temperature < 80:
            return
        if cooldown_start is not None and temperature <= 75:
            return
        cooldown_start = cooldown_start or time.monotonic()
        if time.monotonic()-cooldown_start >= 600:
            raise RuntimeError('THERMAL_SAFE_STOP: cooldown exceeded 10 minutes')
        print(f'Thermal cooldown: {temperature} C; waiting 20 seconds (resume <=75 C)', flush=True)
        time.sleep(20)

def main():
    global TASK
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['task_a','field_condition','field_harvest','field_prop'])
    parser.add_argument('--condition-start', type=Path)
    args = parser.parse_args()
    TASK = args.task
    assert (OUT/'input_validation.json').is_file()
    runtime = OUT/'runtime'/TASK
    runtime.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime)
    torch.set_num_threads(4)
    started = time.time()
    if TASK == 'task_a':
        starting_checkpoint = str(OUT/'preserved_inputs/morning_original_last.pt')
        assert len(list(csv.DictReader((OUT/'task_a/results.csv').open()))) == 11
        if not (OUT/'task_a/weights/last.pt').exists():
            shutil.copy2(OUT/'preserved_inputs/morning_original_last.pt', OUT/'task_a/weights/last.pt')
        model = YOLO(str(OUT/'preserved_inputs/morning_original_last.pt'))
        assert model.ckpt['epoch'] == 10 and model.ckpt.get('optimizer') is not None
        def record_resume(trainer):
            assert trainer.start_epoch == 11 and trainer.epochs == 20
            assert trainer.save_dir.resolve() == (OUT/'task_a').resolve()
            record = {'start_epoch_zero_based':trainer.start_epoch,'next_epoch_one_based':trainer.start_epoch+1,'total_epochs':trainer.epochs,'optimizer_type':type(trainer.optimizer).__name__,'optimizer_state_entries':len(trainer.optimizer.state),'optimizer_lrs':[group['lr'] for group in trainer.optimizer.param_groups],'train_images':len(trainer.train_loader.dataset),'save_dir':str(trainer.save_dir),'data':trainer.args.data}
            (OUT/'resume_verified.json').write_text(json.dumps(record,indent=2)+'\n')
        model.add_callback('on_train_start', record_resume)
        model.add_callback('on_train_batch_start', thermal_guard)
        model.train(resume=True, data=str(OUT/'configs/morning.yaml'), save_dir=str(OUT/TASK), device=0, workers=0, cache=False)
    else:
        starts = {'field_condition': args.condition_start, 'field_harvest': REPO/'continuation_packages/starter_models/harvest_cls_v3.pt', 'field_prop': REPO/'continuation_packages/starter_models/printed_prop_v8.pt'}
        start = starts[TASK]
        assert start is not None and Path(start).is_file()
        starting_checkpoint = str(start)
        assert not (OUT/TASK).exists(), 'Refusing to overwrite an existing field run'
        model = YOLO(str(start))
        model.add_callback('on_train_batch_start', thermal_guard)
        common = dict(project=str(OUT), name=TASK, exist_ok=False, epochs=20, patience=6, device=0, workers=0, optimizer='AdamW', lr0=0.00008, lrf=0.1, warmup_epochs=1, amp=False, seed=20261103, deterministic=True, cache=False, plots=True, save=True, save_period=-1)
        if TASK == 'field_harvest':
            model.train(data=str(WORK/'field_dataset/harvest_classify'), imgsz=224, batch=16, hsv_h=0.015, hsv_s=0.2, hsv_v=0.2, degrees=5, translate=0.05, scale=0.1, fliplr=0.5, flipud=0.0, auto_augment=None, erasing=0.0, **common)
        else:
            config = 'condition' if TASK == 'field_condition' else 'prop'
            model.train(data=str(OUT/f'configs/{config}.yaml'), imgsz=640, batch=4, mosaic=0.2, close_mosaic=5, mixup=0.0, degrees=5, translate=0.05, scale=0.1, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.2, hsv_v=0.2, **common)
    run = OUT/TASK
    rows = list(csv.DictReader((run/'results.csv').open()))
    (OUT/f'{TASK}_completion.json').write_text(json.dumps({'task':TASK,'elapsed_seconds':time.time()-started,'result_rows':len(rows),'last_epoch':int(rows[-1]['epoch']),'intended_epochs':20,'early_stopping_allowed':True,'start_model':starting_checkpoint,'status':'TRAINING_COMPLETE'},indent=2)+'\n')
    print(f'{TASK} TRAINING COMPLETE: {len(rows)} result rows', flush=True)

if __name__ == '__main__':
    main()

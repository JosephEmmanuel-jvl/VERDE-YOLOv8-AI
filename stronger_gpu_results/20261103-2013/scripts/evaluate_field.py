"""Frozen baseline-equivalent field evaluation; no threshold tuning or dataset changes."""
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from prepare_inputs import OUT, SOURCE, WORK, sha

REPO = OUT.parents[1]

def detector_metrics(path, task, folder):
    model = YOLO(str(path))
    metrics = model.val(data=str(OUT/f'configs/{task}.yaml'), split='test', imgsz=640, conf=0.001, iou=0.5, device=0, batch=4, workers=0, verbose=False, plots=True, project=str(folder), name=f'{task}_detection_metrics', exist_ok=False)
    report = {'model_sha256':sha(path),'aggregate':{'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map)},'per_class':{model.names[int(class_id)]:{'precision':float(metrics.box.p[index]),'recall':float(metrics.box.r[index]),'map50':float(metrics.box.ap50[index]),'map50_95':float(metrics.box.all_ap[index].mean())} for index,class_id in enumerate(metrics.box.ap_class_index)}}
    report['note'] = 'Standard full-image detection metrics. Printed-prop primary comparison uses preserved tiled workflow separately.'
    matrix = metrics.confusion_matrix.matrix
    report['confusion_matrix'] = matrix.tolist()
    report['confusion_matrix_operating_point'] = {'confidence':0.001,'matching_iou':0.45,'nms_iou':0.50,'rows':'predicted including background','columns':'truth including background'}
    report['fixed_confidence_counts'] = {model.names[index]:{'tp':int(matrix[index,index]),'fp':int(matrix[index,:].sum()-matrix[index,index]),'fn':int(matrix[:,index].sum()-matrix[index,index])} for index in range(len(model.names))}
    del model, metrics
    gc.collect()
    torch.cuda.empty_cache()
    return report

def enrich(report):
    rows = report['condition']['predictions']
    classes = ['Healthy','Yellowing','Wilting']
    report['condition']['top_prediction_per_class_counts'] = {}
    for name in classes:
        tp = sum(row['truth']==name and row['prediction']==name for row in rows)
        fp = sum(row['truth']!=name and row['prediction']==name for row in rows)
        fn = sum(row['truth']==name and row['prediction']!=name for row in rows)
        report['condition']['top_prediction_per_class_counts'][name] = {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
    negative = [row for row in report['prop']['predictions'] if row['image'].startswith('set-04-')]
    report['prop']['no_prop_frames'] = len(negative)
    report['prop']['no_prop_false_positive_boxes'] = sum(row['fp'] for row in negative)
    report['prop']['no_prop_frames_with_false_positives'] = sum(row['fp']>0 for row in negative)
    return report

def examples(report, paths, folder):
    destination = folder/'annotated_examples'
    destination.mkdir()
    prop_rows = report['prop']['predictions']
    positive = [row for row in prop_rows if row['truth_count']]
    negative = [row for row in prop_rows if not row['truth_count']]
    selected = [positive[0],max(positive,key=lambda row:row['fn']+row['fp']),negative[0],max(negative,key=lambda row:row['fp'])]
    seen = set()
    for row in selected:
        if row['image'] in seen:
            continue
        seen.add(row['image'])
        image = cv2.imread(str(WORK/'field_dataset/prop_detect/images/test'/row['image']))
        for detection in row['detections']:
            x1,y1,x2,y2 = [int(round(value)) for value in detection['box']]
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),4)
            cv2.putText(image,f"Printed prop {detection['confidence']:.2f}",(x1,max(30,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(image,f"Fixed-demo test: TP {row['tp']} FP {row['fp']} FN {row['fn']}",(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3)
        cv2.imwrite(str(destination/('prop_'+row['image'])), image)
    condition = YOLO(str(paths['condition']))
    picks = []
    for name in ['Healthy','Yellowing','Wilting']:
        rows = [r for r in report['condition']['predictions'] if r['truth']==name]
        picks.append(next((r for r in rows if not r['correct']),rows[0]))
    for row in picks:
        path = WORK/'field_dataset/condition_detect/images/test'/row['image']
        prediction = condition.predict(str(path),imgsz=640,conf=0.1,iou=0.5,device=0,verbose=False)[0]
        cv2.imwrite(str(destination/('condition_'+row['image'])),prediction.plot())
    del condition
    gc.collect()
    torch.cuda.empty_cache()
    harvest = YOLO(str(paths['harvest']))
    for name in ['not_ready','ready']:
        rows = [r for r in report['harvest']['predictions'] if r['truth']==name]
        row = next((r for r in rows if not r['correct']),rows[0])
        path = WORK/'field_dataset/harvest_classify/test'/name/row['image']
        prediction = harvest.predict(str(path),imgsz=640,device=0,verbose=False)[0]
        cv2.imwrite(str(destination/('harvest_'+row['image'])),prediction.plot())
    del harvest
    gc.collect()
    torch.cuda.empty_cache()

def main():
    runtime = OUT/'runtime/field_evaluation'
    runtime.mkdir(parents=True,exist_ok=True)
    os.chdir(runtime)
    torch.set_num_threads(4)
    original_script = SOURCE/'scripts/benchmark_current_models.py'
    preserved_script = OUT/'scripts/preserved_benchmark_current_models.py'
    if not preserved_script.exists():
        shutil.copy2(original_script,preserved_script)
    assert sha(original_script)==sha(preserved_script)
    options = {'baseline':{'condition':OUT/'preserved_inputs/morning_original_best.pt','harvest':REPO/'continuation_packages/starter_models/harvest_cls_v3.pt','prop':REPO/'continuation_packages/starter_models/printed_prop_v8.pt'},'candidate':{'condition':OUT/'field_condition/weights/best.pt','harvest':OUT/'field_harvest/weights/best.pt','prop':OUT/'field_prop/weights/best.pt'}}
    final = {'preserved_baseline':json.loads((OUT/'preserved_inputs/baseline-v2.json').read_text()),'runs':{}}
    for name,paths in options.items():
        folder = OUT/f'evaluations/field_{name}'
        command = [sys.executable,str(preserved_script),'--dataset',str(WORK/'field_dataset'),'--condition',str(paths['condition']),'--harvest',str(paths['harvest']),'--prop',str(paths['prop']),'--output',str(folder),'--device','0']
        print('RUN',name,flush=True)
        subprocess.run(command,check=True,cwd=runtime)
        report = enrich(json.loads((folder/'baseline-report.json').read_text()))
        report['hashes'] = {task:sha(path) for task,path in paths.items()}
        report['condition_detection_metrics'] = detector_metrics(paths['condition'],'condition',folder)
        report['prop_fullframe_detection_metrics'] = detector_metrics(paths['prop'],'prop',folder)
        examples(report,paths,folder)
        (folder/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        final['runs'][name] = report
        (OUT/'field_comparison.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({name:{task:{key:value for key,value in report[task].items() if key!='predictions'} for task in ['condition','harvest','prop']} for name,report in final['runs'].items()},indent=2),flush=True)

if __name__ == '__main__':
    main()

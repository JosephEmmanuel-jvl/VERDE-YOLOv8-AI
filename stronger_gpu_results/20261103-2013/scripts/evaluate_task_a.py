"""Preserved comparison workflow, relocated without changing originals; adds per-class P/R."""
import gc
import importlib.util
import json
import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO
from prepare_inputs import sha, SOURCE, OUT, WORK

def evaluate(model_path, name):
    folder = OUT/'evaluations/task_a'/name
    folder.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(model_path))
    assert list(model.names.values()) == ['Healthy','Yellowing','Wilting']
    metrics = model.val(data=str(OUT/'configs/morning.yaml'), split='val', imgsz=640, conf=0.001, iou=0.50, device=0, batch=4, workers=0, verbose=False, plots=True, project=str(folder), name='validation', exist_ok=False)
    report = {'model':str(model_path),'sha256':sha(model_path),'aggregate':{'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map)},'per_class':{}}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        report['per_class'][model.names[int(class_id)]] = {'precision':float(metrics.box.p[index]),'recall':float(metrics.box.r[index]),'map50':float(metrics.box.ap50[index]),'map50_95':float(metrics.box.all_ap[index].mean())}
    if getattr(metrics, 'confusion_matrix', None) is not None:
        matrix = metrics.confusion_matrix.matrix
        report['confusion_matrix'] = matrix.tolist()
        report['confusion_matrix_operating_point'] = {'confidence':0.001,'matching_iou':0.45,'nms_iou':0.50,'rows':'predicted including final background','columns':'truth including final background'}
        report['fixed_confidence_counts'] = {model.names[index]:{'tp':int(matrix[index,index]),'fp':int(matrix[index,:].sum()-matrix[index,index]),'fn':int(matrix[:,index].sum()-matrix[index,index])} for index in range(len(model.names))}
    original_script = SOURCE/'morning-yellowing-training/scripts/compare_candidate.py'
    copied_script = OUT/'scripts/preserved_compare_candidate.py'
    if not copied_script.exists():
        shutil.copy2(original_script,copied_script)
    assert sha(original_script)==sha(copied_script)
    spec = importlib.util.spec_from_file_location('preserved_comparison', copied_script)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    comparison.DATA = WORK/'morning_dataset'
    report['development_yellowing'] = comparison.test_heldout(model)
    report['limitations'] = ['Outside Yellowing photographs are development probes; independence unconfirmed.', 'Original baseline Condition V2 checkpoint referenced by historical script is not supplied; preserved epoch-8 best is the available checkpoint comparison.', 'Validation is not fresh independent Pi-camera/full-pipeline validation.']
    examples = sorted((WORK/'morning_dataset/heldout').glob('YEL-*.png'))[:3]
    if examples:
        model.predict([str(p) for p in examples], imgsz=640, conf=0.1, iou=0.5, device=0, save=True, project=str(folder), name='development_examples', verbose=False)
    (folder/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    del model, metrics
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'name':name, 'per_class':report['per_class']},indent=2),flush=True)
    return report

def main():
    os.chdir(OUT/'runtime/task_a')
    torch.set_num_threads(4)
    paths = {'preserved_best':OUT/'preserved_inputs/morning_original_best.pt','continued_best':OUT/'task_a/weights/best.pt','continued_last':OUT/'task_a/weights/last.pt'}
    reports = {name:evaluate(path,name) for name,path in paths.items()}
    # Choose on existing validation only, before any locked fixed-demo test evaluation.
    # Yellowing AP50-95 is the primary ranking; Wilting AP50-95 breaks ties.
    chosen = max(['preserved_best','continued_best','continued_last'], key=lambda name:(reports[name]['per_class']['Yellowing']['map50_95'],reports[name]['per_class']['Wilting']['map50_95']))
    summary = {'models':reports,'selection_rule':'Highest Yellowing validation AP50-95; Wilting AP50-95 tie-break. No fixed-demo test selection.','selected':chosen,'selected_path':str(paths[chosen]),'selected_sha256':sha(paths[chosen])}
    (OUT/'task_a_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SELECTED TASK A INPUT:',summary['selected_path'],flush=True)

if __name__ == '__main__':
    main()

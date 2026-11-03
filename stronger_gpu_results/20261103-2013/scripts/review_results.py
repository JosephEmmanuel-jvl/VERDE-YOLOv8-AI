"""Summarize completed experiments without changing models, data, thresholds, or test protocol."""
import json
from pathlib import Path
from PIL import Image

OUT=Path(__file__).resolve().parents[1]
field=json.loads((OUT/'field_comparison.json').read_text())
a=json.loads((OUT/'task_a_comparison.json').read_text())
original=field['preserved_baseline']
baseline=field['runs']['baseline']
candidate=field['runs']['candidate']
comparison={}
for task,keys in [('condition',['truth','prediction','correct']),('harvest',['truth','prediction','correct']),('prop',['truth_count','detection_count','tp','fp','fn'])]:
    old={row['image']:row for row in original[task]['predictions']}
    new={row['image']:row for row in baseline[task]['predictions']}
    assert set(old)==set(new),(task,'Baseline image inventory differs')
    differences=[{'image':name,'preserved':{key:old[name][key] for key in keys},'reproduced':{key:new[name][key] for key in keys}} for name in old if any(old[name][key]!=new[name][key] for key in keys)]
    comparison[task]={'images':len(old),'discrete_prediction_differences':differences,'exact_discrete_reproduction':not differences}
(OUT/'baseline_reproduction.json').write_text(json.dumps(comparison,indent=2)+'\n')

harvest={}
for run_name,report in [('baseline',baseline),('candidate',candidate)]:
    rows=report['harvest']['predictions']
    harvest[run_name]={}
    for name in ['not_ready','ready']:
        tp=sum(row['truth']==name and row['prediction']==name for row in rows)
        fp=sum(row['truth']!=name and row['prediction']==name for row in rows)
        fn=sum(row['truth']==name and row['prediction']!=name for row in rows)
        harvest[run_name][name]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
(OUT/'harvest_per_class_review.json').write_text(json.dumps(harvest,indent=2)+'\n')

review={
    'task_a_yellowing_focused':{'path':'task_a/weights/last.pt','sha256':a['models']['continued_last']['sha256'],'role':'YELLOWING_FOCUSED_DEVELOPMENT_CANDIDATE','reason':'Highest Yellowing validation AP50-95; Healthy AP and Wilting AP regress relative to preserved best. This was the pre-test field adaptation starting point.'},
    'task_a_macro_ap_alternative':{'path':'task_a/weights/best.pt','sha256':a['models']['continued_best']['sha256'],'role':'MACRO_AP_WILTING_DEVELOPMENT_ALTERNATIVE','reason':'Higher Wilting and macro AP; Yellowing AP nearly unchanged versus preserved best and Healthy recall lower.'},
    'field_condition':{'path':'field_condition/weights/best.pt','sha256':candidate['hashes']['condition'],'role':'PREFERRED_FRESH_FIXED_DEMO_VALIDATION_CANDIDATE','reason':'Locked fixed-demo top-prediction accuracy 100% versus 77.92%; all three classes 100%. Same six plants and documented validation/test duplicate prevent generalization or acceptance claims.'},
    'field_harvest':{'path':'field_harvest/weights/best.pt','sha256':candidate['hashes']['harvest'],'role':'DEVELOPMENT_REVIEW_ONLY_NO_PROMOTION','reason':'Accuracy rises 50% to64.58% and false-ready predictions fall40 to1, but ready recall falls40/120 to36/120:84 ready examples missed. Preserve both models; no promotion on aggregate accuracy. Training224/evaluation640 preprocessing mismatch is a limitation.'},
    'field_prop':{'preferred_path':'continuation_packages/starter_models/printed_prop_v8.pt','sha256':baseline['hashes']['prop'],'role':'PREFER_PRESERVED_BASELINE_FOR_FUTURE_VALIDATION','rejected_adaptation_path':'field_prop/weights/best.pt','rejected_adaptation_sha256':candidate['hashes']['prop'],'reason':'New adaptation regresses tiled F1 from0.8276 to0.2787; TP60 to17, FP5 to25, FN20 to63. No-prop false positives5 to22. Keep failed run as evidence, do not promote.'},
    'deployment_authorized':False,'independent_acceptance_ready':False,
    'next_action':'Fresh isolated Pi-camera/full-pipeline validation under later authorization. Pre-register input-size/tiling alignment and fresh held-out groups before any further Harvest/prop tuning; no further training in this task.',
    'possible_contributors_not_proven_causes':['Harvest trained224 but preserved benchmark uses640.','Printed-prop fine-tuning used resized full frames while primary benchmark uses640-pixel tiles. These preprocessing differences may contribute to failures; this experiment does not prove causation.']
}
(OUT/'RECOMMENDATION_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')

images=[]
for path in sorted(OUT.rglob('*')):
    if not path.is_file() or path.suffix.lower() not in {'.jpg','.jpeg','.png'} or any(part in {'_work','runtime','__pycache__'} for part in path.relative_to(OUT).parts):
        continue
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        images.append({'path':path.relative_to(OUT).as_posix(),'width':image.width,'height':image.height,'format':image.format})
(OUT/'image_output_validation.json').write_text(json.dumps({'status':'PASS','decoded_images':len(images),'malformed_images':0,'images':images},indent=2)+'\n')
print(json.dumps({'baseline_exact_discrete_reproduction':{task:value['exact_discrete_reproduction'] for task,value in comparison.items()},'output_images_decoded':len(images),'recommendations':{key:value['role'] for key,value in review.items() if isinstance(value,dict) and 'role'in value}},indent=2))

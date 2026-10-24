"""Re-evaluate preserved checkpoints on copied historical test images using CPU."""
from __future__ import annotations
import hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import torch
from ultralytics import YOLO

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'evidence'
DATA=REPO/'dataset'/'condition3_evidence_20261024'/'data.yaml'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if not (OUT/'INTAKE_AUDIT.json').is_file():raise RuntimeError('Complete preservation audit first')
    runtime={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':importlib.metadata.version('torchvision'),'ultralytics':importlib.metadata.version('ultralytics'),'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'device':'cpu','threads':4,'seed_override':None,'augmentation':False,'pinned_repository_ultralytics':'8.4.174','runtime_version_difference':'8.4.184 installed; report separately from published metrics'}
    (OUT/'RUNTIME.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    names={0:'Healthy',1:'Yellowing',2:'Wilting'}
    for stem in ['verde_condition3_best_validation','verde_condition3_final_gpu_resume']:
        checkpoint=REPO/'models'/(stem+'.pt')
        output=OUT/(stem+'_cpu_retest.json')
        if output.exists():print('Existing result retained',output,flush=True);continue
        model=YOLO(str(checkpoint));assert model.names==names,model.names
        started=time.perf_counter()
        metric=model.val(data=str(DATA),split='test',imgsz=416,batch=4,device='cpu',workers=0,conf=0.001,iou=0.7,max_det=300,plots=True,save_json=True,save_txt=True,save_conf=True,project=str(REPO/'runs'/'evidence_20261024'),name=stem,exist_ok=False,verbose=True)
        per_class=[]
        for i,cls in enumerate(metric.box.ap_class_index):
            p,r,ap50,ap=metric.box.class_result(i)
            per_class.append({'class_id':int(cls),'class':names[int(cls)],'precision':float(p),'recall':float(r),'f1':float(2*p*r/(p+r)) if p+r else 0.,'ap50':float(ap50),'ap50_95':float(ap),'note':'P/R/F1 use Ultralytics best mean-F1 point on the evaluated test curve; descriptive, not a locked deployment threshold. See separate conf=0.25 report.'})
        report={'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'class_names':model.names,'parameters':sum(p.numel() for p in model.model.parameters()),'model_yaml':model.model.yaml,'dataset_yaml':str(DATA),'split':'test','images':200,'imgsz':416,'batch':4,'device':'cpu','confidence_for_ap':0.001,'nms_iou':0.7,'max_det':300,'runtime':runtime,'metrics':{k:float(v) for k,v in metric.results_dict.items()},'per_class':per_class,'speed_ms_per_image':metric.speed,'wall_seconds':time.perf_counter()-started,'save_dir':str(metric.save_dir),'independent_acceptance':False,'decision':'EXPERIMENTAL_BASELINE_ONLY'}
        output.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8');print(json.dumps(report,indent=2,default=str),flush=True)
    (OUT/'EVALUATION_SAVEPOINT.json').write_text(json.dumps({'phase':'HISTORICAL_INTERNAL_RETEST_COMPLETE','training_started':False,'new_acceptance_set_evaluated':False,'model_files_modified':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()

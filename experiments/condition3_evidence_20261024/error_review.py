"""Fixed-threshold error analysis and comparisons using saved predictions only."""
from __future__ import annotations
import csv,json,shutil
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent; E=HERE/'evidence'; REPO=HERE.parents[1]
NAMES={0:'Healthy',1:'Yellowing',2:'Wilting'}

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def boxes(p,pred=False):
    if not p.is_file():return []
    result=[]
    for s in p.read_text().splitlines():
        if not s.strip():continue
        vals=list(map(float,s.split()));c=int(vals[0]);x,y,w,h=vals[1:5];conf=vals[5] if pred else 1.
        if pred and conf<.25:continue
        result.append({'class':c,'xyxy':[x-w/2,y-h/2,x+w/2,y+h/2],'confidence':conf})
    return result
def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0
def match(gt,pred):
    pairs=sorted([(iou(g['xyxy'],p['xyxy']),gi,pi) for gi,g in enumerate(gt) for pi,p in enumerate(pred) if g['class']==p['class'] and iou(g['xyxy'],p['xyxy'])>=.5],reverse=True)
    usedg=set();usedp=set();found=[]
    for score,gi,pi in pairs:
        if gi in usedg or pi in usedp:continue
        usedg.add(gi);usedp.add(pi);found.append((gi,pi,score))
    return found,usedg,usedp
def draw_panel(image,bb,title):
    im=image.copy();d=ImageDraw.Draw(im);w,h=im.size
    for b in bb:
        x1,y1,x2,y2=b['xyxy'];colour={0:'#00aaee',1:'#f0b000',2:'#ff5555'}[b['class']]
        d.rectangle((x1*w,y1*h,x2*w,y2*h),outline=colour,width=max(2,w//300))
        d.text((max(0,x1*w),max(0,y1*h)),NAMES[b['class']]+f" {b['confidence']:.2f}",fill='white',stroke_width=1,stroke_fill='black')
    im.thumbnail((700,580));canvas=Image.new('RGB',(720,640),'white');canvas.paste(im,((720-im.width)//2,50));ImageDraw.Draw(canvas).text((10,10),title,fill='black');return canvas

def main():
    manifest=[r for r in read(E/'EXISTING_CONDITION_DATA_MANIFEST.csv') if r['split']=='test']
    metricrows=[];details=[];results={}
    out=E/'error_review';out.mkdir(exist_ok=True)
    for stem in ['verde_condition3_best_validation','verde_condition3_final_gpu_resume']:
        report_path=E/(stem+'_cpu_retest.json');report=json.loads(report_path.read_text())
        for c in report['per_class']:c['note']='P/R/F1 use Ultralytics best mean-F1 point on the evaluated test curve; descriptive, not a locked deployment threshold. See separate conf=0.25 report.'
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        preds=Path(report['save_dir'])/'labels';gt_counts=Counter();tp=Counter();fp=Counter();fn=Counter();cases=[]
        for r in manifest:
            gt=boxes(Path(r['label']));pd=boxes(preds/(Path(r['image']).stem+'.txt'),True);matches,ug,up=match(gt,pd)
            for b in gt:gt_counts[b['class']]+=1
            for gi,pi,score in matches:tp[gt[gi]['class']]+=1
            for i,b in enumerate(gt):
                if i not in ug:fn[b['class']]+=1
            for i,b in enumerate(pd):
                if i not in up:fp[b['class']]+=1
            mismatches=[]
            for g in gt:
                for p in pd:
                    if g['class']!=p['class'] and iou(g['xyxy'],p['xyxy'])>=.5:mismatches.append(NAMES[g['class']]+'->'+NAMES[p['class']])
            d={'checkpoint':stem,'candidate_id':r['candidate_id'],'class':r['class'],'image_sha256':r['inference_image_sha256'],'group_id':r['group_id'],'ground_truth_boxes':len(gt),'predictions_conf_ge_025':len(pd),'true_positives_iou_05':len(matches),'false_positives':len(pd)-len(matches),'missed_boxes':len(gt)-len(matches),'cross_class_overlap':';'.join(mismatches),'predictions':json.dumps(pd,separators=(',',':')),'image_path':r['image'],'label_path':r['label']}
            details.append(d);cases.append((d,r,gt,pd))
        result=[]
        for cid,name in NAMES.items():
            p=tp[cid]/(tp[cid]+fp[cid]) if tp[cid]+fp[cid] else 0;r=tp[cid]/gt_counts[cid] if gt_counts[cid] else 0
            result.append({'checkpoint':stem,'class':name,'conf':.25,'iou_match':.5,'ground_truth_boxes':gt_counts[cid],'true_positive':tp[cid],'false_positive':fp[cid],'false_negative':fn[cid],'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0})
        metricrows.extend(result);results[stem]=result
        examples=[]
        for cls in NAMES.values():
            ranked=sorted([x for x in cases if x[0]['class']==cls],key=lambda x:(-(x[0]['false_positives']+x[0]['missed_boxes']),x[0]['candidate_id']))
            examples.extend(ranked[:3])
            successes=[x for x in ranked if x[0]['false_positives']==0 and x[0]['missed_boxes']==0]
            if successes:examples.append(successes[0])
        saved=[]
        for n,(d,r,gt,pd) in enumerate(examples):
            with Image.open(r['image']) as im:image=im.convert('RGB')
            sheet=Image.new('RGB',(1440,690),'#eeeeee');sheet.paste(draw_panel(image,gt,'Historical ground truth (unchanged)'),(0,50));sheet.paste(draw_panel(image,pd,'Prediction conf >=0.25; confidence shown'),(720,50))
            ImageDraw.Draw(sheet).text((10,10),f"{r['candidate_id']} | {r['class']} | FP={d['false_positives']} FN={d['missed_boxes']} | historical test only",fill='black')
            dest=out/(stem+f'_{n+1:02}.jpg');sheet.save(dest,quality=92);saved.append(dest)
        for p in sorted(Path(report['save_dir']).glob('*curve.png'))+sorted(Path(report['save_dir']).glob('confusion_matrix*.png')):
            shutil.copy2(p,out/(stem+'_'+p.name))
    write(E/'FIXED_CONFIDENCE_CLASS_METRICS.csv',metricrows);write(E/'TEST_IMAGE_ERRORS.csv',details)
    (E/'FIXED_THRESHOLD_SUMMARY.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()

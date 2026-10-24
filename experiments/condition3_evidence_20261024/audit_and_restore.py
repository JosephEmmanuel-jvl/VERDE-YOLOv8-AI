"""Audit original evidence and prepare isolated CPU evaluation copies, never training."""
from __future__ import annotations
import csv, hashlib, json, platform, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'evidence'
SOURCE = Path(r'D:\VERDE_AI_DATA\verde_v2')
DECODED = Path(r'D:\VERDE_AI_DATA\verde_v2_anylabeling')
DATASET = REPO / 'dataset' / 'condition3_evidence_20261024'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csvwrite(p, rows, fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'INTAKE_AUDIT.json').exists():raise RuntimeError('Audit already exists; do not redo intake')
    protected=[]
    saved_protection=OUT/'PROTECTED_INPUT_CHECKSUMS_BEFORE.csv'
    if saved_protection.exists():protected=csvread(saved_protection)
    roots=[SOURCE,DECODED,Path(r'D:\Documents\Codex\VERDE_CANONICAL_ANNOTATION_WORKING_SET_20261023')]
    v2=Path(r'D:\VERDE_V2_BUILD\VERDE_APPROVED_CLASS_POOL_V2_20261004.zip')
    for root in ([] if protected else roots):
        for p in sorted(root.rglob('*')):
            if p.is_file():protected.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    if v2.is_file() and not any(r['path']==str(v2) for r in protected):protected.append({'path':str(v2),'bytes':v2.stat().st_size,'sha256':sha(v2)})
    for p in sorted((REPO/'models').glob('*.pt')):
        dest=REPO/'runs'/'rollback_20261024'/p.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(p,dest)
        if sha(p)!=sha(dest):raise RuntimeError('Rollback copy mismatch')
        if not any(r['path']==str(p) for r in protected):protected.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    csvwrite(OUT/'PROTECTED_INPUT_CHECKSUMS_BEFORE.csv',protected)
    print('Protected input hashes saved:',len(protected),flush=True)
    splits=csvread(SOURCE/'metadata'/'V2_SPLIT_ASSIGNMENTS.csv')
    provenance={r['sha256']:r for r in csvread(SOURCE/'metadata'/'V2_TRAINING_ELIGIBLE_MANIFEST.csv')}
    records=[]; skipped=[]; errors=[]; identity_differences=[]; groups=defaultdict(set); sources=defaultdict(set)
    for row in splits:
        label=SOURCE/row['export_label_path']
        lines=[s.split() for s in label.read_text(encoding='utf-8').splitlines() if s.strip()]
        ids={int(s[0]) for s in lines}
        if not ids or ids-{0,1,2}:
            skipped.append({'candidate_id':row['candidate_id'],'split':row['split'],'class':row['class'],'reason':'EXCLUDED_NON_CONDITION_LABELS'});continue
        original=SOURCE/row['export_image_path']
        image=original
        if image.suffix.lower() in {'.heic','.heif'}:image=DECODED/row['split']/(original.stem+'.jpg')
        digest=sha(original); review_digest=sha(image)
        if digest!=row['sha256']:identity_differences.append({'candidate_id':row['candidate_id'],'canonical_ledger_sha256':row['sha256'],'historical_binary_sha256':digest,'status':'HISTORICAL_BINARY_DIFFERS_CANONICAL_FIVE_TRANSFER_CASES','used_for':'HISTORICAL_BASELINE_RETEST_ONLY'})
        with Image.open(image) as im:
            im.load(); width,height=im.size
        for parts in lines:
            if len(parts)!=5:errors.append('LABEL_FIELDS:'+row['candidate_id']);continue
            cid,*vals=map(float,parts)
            if cid not in {0.,1.,2.} or not all(0<=v<=1 for v in vals) or min(vals[2:])<=0:errors.append('LABEL_GEOMETRY:'+row['candidate_id'])
        imgout=DATASET/'images'/row['split']/image.name
        lblout=DATASET/'labels'/row['split']/label.name
        imgout.parent.mkdir(parents=True,exist_ok=True);lblout.parent.mkdir(parents=True,exist_ok=True)
        if not imgout.exists():shutil.copy2(image,imgout)
        if not lblout.exists():shutil.copy2(label,lblout)
        if sha(imgout)!=review_digest or sha(lblout)!=sha(label):raise RuntimeError('COPY_HASH')
        prov=provenance[row['sha256']]
        records.append({'candidate_id':row['candidate_id'],'class':row['class'],'class_id':row['class_id'],'split':row['split'],'original_sha256':digest,'inference_image_sha256':review_digest,'original_image':str(original),'image':str(imgout),'label':str(lblout),'label_sha256':sha(label),'width':width,'height':height,'box_count':len(lines),'group_id':row['group_id'],'grouping_confidence':row['grouping_confidence'],'source':row['source'],'source_dataset':row['source_dataset'],'source_url':prov['source_url'],'license':prov['license'],'provenance_status':prov['provenance_status'],'reviewer':'HISTORICAL_LABELS_STRUCTURAL_AUDIT_ONLY'})
        groups[row['group_id']].add(row['split']);sources[row['source_dataset']].add(row['split'])
    csvwrite(OUT/'EXISTING_CONDITION_DATA_MANIFEST.csv',records)
    csvwrite(OUT/'EXCLUDED_NON_CONDITION_RECORDS.csv',skipped)
    csvwrite(OUT/'HISTORICAL_CANONICAL_IDENTITY_DIFFERENCES.csv',identity_differences)
    hash_splits=defaultdict(set)
    for r in records:hash_splits[r['inference_image_sha256']].add(r['split'])
    group_rows=[{'group_id':k,'splits':';'.join(sorted(v)),'cross_split':len(v)>1,'images':sum(r['group_id']==k for r in records)} for k,v in groups.items()]
    csvwrite(OUT/'EXISTING_GROUP_SPLIT_AUDIT.csv',group_rows)
    source_rows=[{'source_dataset':k,'splits':';'.join(sorted(v)),'images':sum(r['source_dataset']==k for r in records),'cross_split':len(v)>1} for k,v in sources.items()]
    csvwrite(OUT/'EXISTING_SOURCE_SPLIT_AUDIT.csv',source_rows)
    if errors:raise RuntimeError(errors)
    (DATASET/'data.yaml').write_text(f'path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Healthy\n  1: Yellowing\n  2: Wilting\n',encoding='utf-8')
    samples=[]
    for cls in ('Healthy','Yellowing','Wilting'):
        available=sorted([r for r in records if r['class']==cls],key=lambda r:(r['split']!='test',r['group_id'],r['candidate_id']))
        seen=set()
        for r in available:
            if r['group_id'] in seen:continue
            seen.add(r['group_id']);samples.append(r)
            if len(seen)>=6:break
    preview_dir=OUT/'historical_label_audit';preview_dir.mkdir(exist_ok=True)
    for r in samples:
        with Image.open(r['image']) as im:im=ImageOps.exif_transpose(im).convert('RGB')
        w,h=im.size;draw=ImageDraw.Draw(im)
        for line in Path(r['label']).read_text().splitlines():
            if not line.strip():continue
            cid,x,y,bw,bh=map(float,line.split())
            draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='red',width=max(2,w//300))
        im.thumbnail((900,650)); panel=Image.new('RGB',(940,720),'white');panel.paste(im,(20,55));d=ImageDraw.Draw(panel)
        d.text((20,10),f"{r['candidate_id']} | {r['class']} | {r['split']} | boxes={r['box_count']}",fill='black')
        panel.save(preview_dir/(r['candidate_id'].replace('/','_')+'.jpg'),quality=92)
    csvwrite(OUT/'HISTORICAL_LABEL_REVIEW_SAMPLE.csv',samples)
    for i in range(0,len(samples),6):
        sheet=Image.new('RGB',(1880,2160),'#eeeeee')
        for j,r in enumerate(samples[i:i+6]):
            with Image.open(preview_dir/(r['candidate_id'].replace('/','_')+'.jpg')) as im:sheet.paste(im,((j%2)*940,(j//2)*720))
        sheet.save(preview_dir/f'AUDIT_SHEET_{i//6+1:02}.jpg',quality=94)
    summary={'records':len(records),'boxes':sum(r['box_count'] for r in records),'split_counts':dict(Counter(r['split'] for r in records)),'class_counts':dict(Counter(r['class'] for r in records)),'groups':len(groups),'declared_group_cross_split':sum(len(v)>1 for v in groups.values()),'exact_binary_cross_split':sum(len(v)>1 for v in hash_splits.values()),'source_families_cross_split':sum(len(v)>1 for v in sources.values()),'source_independence':'NOT_ESTABLISHED','historical_vs_canonical_binary_differences':len(identity_differences),'independent_acceptance_images':0,'training_started':False,'dataset_yaml':str(DATASET/'data.yaml'),'protected_files':len(protected),'errors':errors,'python':sys.version,'platform':platform.platform()}
    (OUT/'INTAKE_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

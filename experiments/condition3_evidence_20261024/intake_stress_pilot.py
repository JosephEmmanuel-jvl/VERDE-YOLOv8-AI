"""Acquire a small rights-documented public research pilot into quarantine."""
from __future__ import annotations
import csv, hashlib, json, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
OUT=HERE/'evidence'/'new_source_pilot'
RAW=HERE.parents[1]/'runs'/'new_source_pilot_originals'
URL='https://data.mendeley.com/datasets/294zk6k5wf/2'
API='https://data.mendeley.com/public-api/datasets/294zk6k5wf/files?folder_id=9d2b3fb6-a221-427c-a7a6-e7fdd6ab3e51&version=2&$start=0&$limit=1000'

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'VERDE-evidence-audit/1.0','Accept':'application/vnd.mendeley-public-dataset.1+json'})
    return urllib.request.urlopen(req,timeout=60).read()

def main():
    OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(get(API));(OUT/'original_file_listing.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (OUT/'source_page.html').write_bytes(get(URL))
    def acquire(item):
        p=RAW/item['filename']
        if not p.exists():p.write_bytes(get(item['content_details']['download_url']))
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=item['content_details']['sha256_hash']:raise ValueError('Publisher checksum mismatch '+p.name)
        with Image.open(p) as im:im.load();w,h=im.size
        return {'candidate_id':'STRESS-V2-'+p.stem,'filename':p.name,'sha256':digest,'publisher_sha256_match':True,'width':w,'height':h,'source_url':URL,'source_file_id':item['id'],'binary_url':item['content_details']['download_url'],'owner':'Georgios Fevgas; Thomas Lagkas; Vasileios Argyriou; Panagiotis Sarigiannidis','license':'CC BY 4.0','group_id':'FEVGAS_LAB_NORMAL_PLANT' if p.stem.startswith('Normal') else 'FEVGAS_LAB_NONIRRIGATED_PLANT','group_basis':'Original dataset describes exactly two plants, irrigated and non-irrigated; filenames Normal/Stress','assignment':'ACCEPTANCE_CANDIDATE_QUARANTINE_NO_TRAINING','reviewer':'CODEX_ONLY_PENDING_VISUAL_REVIEW','visual_class':'UNASSIGNED','path':str(p)}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(acquire,metadata))
    rows.sort(key=lambda r:(r['group_id'],int(r['filename'].rsplit('_',1)[-1].split('.')[0])))
    with (OUT/'PILOT_BINARY_INVENTORY.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    for offset in range(0,len(rows),8):
        sheet=Image.new('RGB',(1600,1600),'white');d=ImageDraw.Draw(sheet)
        for i,r in enumerate(rows[offset:offset+8]):
            with Image.open(r['path']) as im:
                im=im.convert('RGB');im.thumbnail((780,350));x=(i%2)*800;y=(i//2)*400
                sheet.paste(im,(x,y+40));d.text((x+5,y+10),r['filename']+' | '+r['group_id'],fill='black')
        sheet.save(OUT/f'PILOT_RAW_SHEET_{offset//8+1:02}.jpg',quality=94)
    print(json.dumps({'downloaded':len(rows),'groups':len(set(r['group_id'] for r in rows)),'license':'CC BY 4.0','assignment':'QUARANTINE_NO_TRAINING','path':str(OUT)},indent=2))

if __name__=='__main__':main()

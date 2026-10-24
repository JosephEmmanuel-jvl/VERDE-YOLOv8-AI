"""Extract embedded JPEG streams losslessly from the research pilot originals."""
import csv, hashlib, io, json, re
from pathlib import Path
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
OUT=HERE/'evidence'/'new_source_pilot'
DEST=HERE.parents[1]/'runs'/'new_source_pilot_embedded_rgb'

def end_of_jpeg(b,start):
    pos=start+2
    while pos<len(b)-1:
        if b[pos]!=255:pos+=1;continue
        while pos<len(b)-1 and b[pos+1]==255:pos+=1
        marker=b[pos+1];pos+=2
        if marker==217:return pos
        if marker==0 or marker==216 or 208<=marker<=215:continue
        if pos+2>len(b):break
        length=int.from_bytes(b[pos:pos+2],'big');pos+=length
    raise ValueError('No JPEG end')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with (OUT/'PILOT_BINARY_INVENTORY.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    out=[]
    for r in rows:
        b=Path(r['path']).read_bytes();streams=[]
        for m in re.finditer(b'\xff\xd8\xff',b):
            start=m.start()
            if start==0:continue
            try:
                end=end_of_jpeg(b,start);blob=b[start:end]
                with Image.open(io.BytesIO(blob)) as im:im.load();w,h=im.size
                if w>=320 and h>=240:streams.append((w*h,start,end,blob,w,h))
            except Exception:continue
        if len(streams)!=1:raise ValueError((r['filename'],'ambiguous embedded RGB candidates',len(streams)))
        _,start,end,blob,w,h=streams[0];dest=DEST/r['filename'];dest.write_bytes(blob)
        out.append({**r,'original_sha256':r['sha256'],'embedded_sha256':hashlib.sha256(blob).hexdigest(),'embedded_path':str(dest),'embedded_start_offset':start,'embedded_end_offset':end,'width':w,'height':h,'extraction':'LOSSLESS_JPEG_STREAM_EXTRACTION_NO_PIXEL_MODIFICATION','reviewer':'CODEX_VISUAL_REVIEW_PENDING','visual_class':'UNASSIGNED'})
    with (OUT/'EMBEDDED_RGB_LINKAGE.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=out[0]);wr.writeheader();wr.writerows(out)
    for offset in range(0,len(out),8):
        sheet=Image.new('RGB',(1600,1600),'white');d=ImageDraw.Draw(sheet)
        for i,r in enumerate(out[offset:offset+8]):
            with Image.open(r['embedded_path']) as im:
                im=im.convert('RGB');im.thumbnail((780,350));x=(i%2)*800;y=(i//2)*400
                sheet.paste(im,(x,y+40));d.text((x+5,y+10),r['filename']+' | embedded RGB | '+r['group_id'],fill='black')
        sheet.save(OUT/f'PILOT_EMBEDDED_RGB_SHEET_{offset//8+1:02}.jpg',quality=94)
    print(json.dumps({'extracted':len(out),'groups':len(set(r['group_id'] for r in out)),'out':str(OUT)},indent=2))

if __name__=='__main__':main()

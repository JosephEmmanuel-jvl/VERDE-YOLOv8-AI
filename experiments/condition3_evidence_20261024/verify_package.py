"""Refresh/verify evidence checksums and verify staged Git blobs byte-for-byte."""
import argparse,hashlib,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');ap.add_argument('--index',action='store_true');args=ap.parse_args()
    manifest=HERE/'CHECKSUMS.sha256'
    if args.refresh:
        files=sorted(p for p in HERE.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts)
        manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(HERE).as_posix()}\n' for p in files),encoding='utf-8')
    rows=[line.split('  ',1) for line in manifest.read_text().splitlines()];bad=[]
    for digest,rel in rows:
        p=HERE/rel
        if sha(p)!=digest:bad.append(rel)
        if args.index:
            path=p.relative_to(REPO).as_posix();blob=subprocess.check_output(['git','-C',str(REPO),'show',':'+path])
            if hashlib.sha256(blob).hexdigest()!=digest:bad.append('INDEX:'+rel)
    print({'checksum_entries':len(rows),'verify_git_index':args.index,'failures':bad})
    if bad:raise SystemExit(1)
if __name__=='__main__':main()

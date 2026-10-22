"""Verify the experiment files with standard Python, without CUDA or ML libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--quick",action="store_true",help="Skip weights, binary states and files larger than20MiB")
    args=parser.parse_args();root=args.root.resolve()
    manifest=json.loads((root/"migration/file_manifest.json").read_text())
    failures=[];checked=0;skipped=0;checked_bytes=0;started=time.monotonic();last=started
    binary={".safetensors",".pt",".bin",".arrow",".so",".whl"}
    for record in manifest["files"]:
        rel=Path(record["path"])
        if rel.is_absolute() or ".." in rel.parts:raise ValueError(f"Unsafe manifest path: {rel}")
        if args.quick and (record["bytes"]>20*2**20 or rel.suffix in binary):
            skipped+=1;continue
        path=root/rel
        if not path.is_file():failures.append({"path":str(rel),"error":"missing"});continue
        if path.stat().st_size!=record["bytes"]:failures.append({"path":str(rel),"error":"size mismatch"});continue
        if sha(path)!=record["sha256"]:failures.append({"path":str(rel),"error":"SHA256 mismatch"});continue
        checked+=1;checked_bytes+=record["bytes"]
        now=time.monotonic()
        if now-last>10:
            print(f"Verified {checked} files, {checked_bytes/2**30:.2f}GiB",flush=True);last=now
    report={"status":"passed" if not failures else "failed","mode":"quick" if args.quick else "full",
        "checked_files":checked,"skipped_files":skipped,"checked_bytes":checked_bytes,
        "seconds":round(time.monotonic()-started,2),"failures":failures}
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 1 if failures else 0


if __name__=="__main__":sys.exit(main())

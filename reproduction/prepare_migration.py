"""Export a handoff ledger, environment snapshots, checksums and a small evidence ZIP.

Never deletes or changes existing experiment results/weights. Does not send data.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"migration"
RUNTIME_DIRS=("artifacts/recheck/eval_env","artifacts/trajectory/author_env","artifacts/trajectory/setup_python312")
GENERATED={"migration/SHA256SUMS","migration/SHA256SUMS.sha256","migration/file_manifest.json",
    "migration/transfer_summary.json","TINA_evidence.zip","TINA_evidence.zip.sha256"}


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def excluded(rel):
    name=rel.as_posix()
    return name in GENERATED or ".git" in rel.parts or "__pycache__" in rel.parts or name.endswith(".pyc") or any(name==p or name.startswith(p+"/") for p in RUNTIME_DIRS)


def project_files():
    for current,dirs,files in os.walk(ROOT):
        dirs[:]=sorted(d for d in dirs if not excluded((Path(current)/d).relative_to(ROOT)) and not (Path(current)/d).is_symlink())
        for name in sorted(files):
            path=Path(current)/name;rel=path.relative_to(ROOT)
            if not excluded(rel) and not path.is_symlink() and path.is_file():yield path


def export_environment():
    code='''import importlib.metadata as m,json,sys,re
from pathlib import Path
names={d.metadata.get("Name") for d in m.distributions()}-{None}
packages={re.sub(r"[-_.]+","-",n).lower():m.version(n) for n in names}
direct={}
for n in names:
    raw=m.distribution(n).read_text("direct_url.json")
    if raw: direct[n]=json.loads(raw)
print(json.dumps({"executable":sys.executable,"resolved_executable":str(Path(sys.executable).resolve()),"prefix":sys.prefix,"base_prefix":sys.base_prefix,"base_executable":sys._base_executable,"python_version":sys.version,"packages":packages,"direct_urls":direct}))'''
    environments={}
    for name,python in {"train":Path("/root/autodl-tmp/envs/tina/bin/python"),
        "eval":ROOT/"artifacts/recheck/eval_env/bin/python", "author_control":ROOT/"artifacts/trajectory/author_env/bin/python"}.items():
        value=json.loads(subprocess.check_output([str(python),"-c",code],text=True))
        environments[name]=value
        (OUT/f"{name}.freeze.txt").write_text("".join(f"{k}=={v}\n" for k,v in sorted(value["packages"].items())))
    base=environments["train"]["packages"]
    diff={k:v for k,v in environments["eval"]["packages"].items() if base.get(k)!=v}
    (OUT/"eval_overrides.txt").write_text("".join(f"{k}=={v}\n" for k,v in sorted(diff.items())))
    write_json(OUT/"environment_snapshot.json",environments)
    conda=Path(environments["train"]["prefix"])/"conda-meta"
    write_json(OUT/"train_conda_packages.json",[json.loads(p.read_text()) for p in sorted(conda.glob("*.json"))])


def export_scores():
    core=json.loads((ROOT/"artifacts/core_suite/results.json").read_text())
    mechanism=json.loads((ROOT/"artifacts/mechanism/results.json").read_text())
    assert core["status"]==mechanism["status"]=="complete"
    combined={**core["models"],**mechanism["models"]}
    baseline=combined["base"]["six_benchmark_mean"]
    labels={"base":"基座","local500":"本地RS3 500步","local850":"本地RS3 850步", "rs2_450":"本地RS2 450步",
        "official500":"作者RS3 500步权重", "official_format_500":"作者格式配方 本地500步", "official_format_850":"作者格式配方 本地850步",
        "label_free_format_500":"去除答案依赖 本地500步", "label_free_format_850":"去除答案依赖 本地850步"}
    registry=[];means=[]
    for model,data in combined.items():
        means.append({"model":model,"label":labels[model],"six_benchmark_mean":data["six_benchmark_mean"],
            "accuracy_percent":data["six_benchmark_mean"]*100,"delta_to_base_pp":(data["six_benchmark_mean"]-baseline)*100})
        for task,score in data["tasks"].items():
            path=Path(score["source"]);directory=ROOT/path.parent
            parquet=sorted(str(p.relative_to(ROOT)) for p in directory.glob("details/**/*.parquet"))
            assert len(parquet)==1 and (ROOT/path).exists()
            summary=json.loads((ROOT/path).read_text())
            assert abs(score["correct"]/score["samples"]-score["accuracy"])<1e-12
            predictions=sorted(str(p.relative_to(ROOT)) for p in directory.glob("*.jsonl"))
            registry.append({"model":model,"label":labels[model],"task":task,"samples":score["samples"],
                "correct":score["correct"],"accuracy":score["accuracy"],"accuracy_percent":score["accuracy"]*100,
                "summary_path":str(path),"parquet_paths":parquet,"prediction_paths":predictions,
                "manifest_path":str(path.parent/"manifest.json"),"manifest":summary["manifest"]})
    assert len(registry)==54 and sum(r["samples"] for r in registry)==9630
    write_json(OUT/"evaluation_registry.json",{"evaluations":54,"predictions":9630,"records":registry})
    write_json(OUT/"final_scores.json",means)
    with (OUT/"final_scores.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(means[0]));writer.writeheader();writer.writerows(means)
    fields=["model","label","task","samples","correct","accuracy","accuracy_percent","summary_path"]
    with (OUT/"task_scores.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(registry)
    return len(registry),sum(r["samples"] for r in registry)


def export_external():
    links=[]
    for current,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d!=".git"]
        for name in dirs+files:
            path=Path(current)/name
            if not path.is_symlink():continue
            target=path.resolve()
            if target.is_relative_to(ROOT):continue
            record={"path":str(path.relative_to(ROOT)),"target":str(target),"exists":path.exists()}
            if name=="debug-core.log" and path.is_file():
                saved=OUT/"external_debug_logs"/target.name;saved.parent.mkdir(exist_ok=True)
                shutil.copy2(target,saved);record["snapshot"]=str(saved.relative_to(ROOT))
            links.append(record)
    write_json(OUT/"external_symlinks.json",links)
    parent=ROOT.parent
    wheels=sorted(parent.glob("*.whl"))+sorted((parent/"wheels").glob("*.whl"))
    wheel_records=[{"path":str(p.relative_to(parent)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in wheels]
    write_json(OUT/"external_wheels.json",wheel_records)
    (OUT/"external_wheels.sha256").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in wheel_records))


def include_evidence(path):
    rel=path.relative_to(ROOT)
    if rel.parts[0]!="artifacts":return True
    if "lighteval_vendor" in rel.parts:return False
    if "ckpts" in rel.parts:
        return path.name in {"trainer_state.json","adapter_audit.json","adapter_config.json","train_results.json","all_results.json"} or path.name.startswith("resources_rank")
    if any(p.startswith("merged_") or p=="temporary_merged" for p in rel.parts):return path.name=="merge_audit.json"
    if path.name in {"tokenizer.json","tokenizer_config.json","special_tokens_map.json"}:return False
    return path.suffix in {".json",".jsonl",".csv",".parquet",".png",".svg",".md",".txt",".yaml",".html",".pdf",".diff"}


def main():
    OUT.mkdir(exist_ok=True)
    export_environment();evaluations,predictions=export_scores();export_external()
    for args,name in [(["git","status","--short"],"git_status.txt"),(["git","rev-parse","HEAD"],"upstream_head.txt"),
        (["git","diff","--binary"],"local_changes.patch")]:
        (OUT/name).write_bytes(subprocess.check_output(args,cwd=ROOT))
    files=[];total=0;last=time.monotonic()
    for path in project_files():
        size=path.stat().st_size
        files.append({"path":str(path.relative_to(ROOT)),"bytes":size,"sha256":digest(path)})
        total+=size
        if time.monotonic()-last>10:
            print(f"Hashed {len(files)} files, {total/2**30:.2f}GiB",flush=True);last=time.monotonic()
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"root_at_creation":str(ROOT),
        "excluded_runtime_dirs":RUNTIME_DIRS,"excluded_other":[".git","__pycache__",*sorted(GENERATED)],
        "file_count":len(files),"bytes":total,"files":files}
    write_json(OUT/"file_manifest.json",manifest)
    (OUT/"SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in files))
    (OUT/"SHA256SUMS.sha256").write_text(f"{digest(OUT/'SHA256SUMS')}  migration/SHA256SUMS\n")
    archive=ROOT/"TINA_evidence.zip";archive_files=[];raw_size=0
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as handle:
        candidates=list(project_files())+[OUT/"file_manifest.json",OUT/"SHA256SUMS",OUT/"SHA256SUMS.sha256"]
        for path in candidates:
            if include_evidence(path):
                handle.write(path,str(path.relative_to(ROOT)));archive_files.append(str(path.relative_to(ROOT)));raw_size+=path.stat().st_size
        archive_manifest={"scope":"Evidence only: no model weights, optimizer state or Python runtime. Full migration copies the project and external environment separately.",
            "file_count":len(archive_files),"uncompressed_bytes":raw_size,"files":archive_files}
        handle.writestr("migration/evidence_contents.json",json.dumps(archive_manifest,indent=2,ensure_ascii=False))
    (ROOT/"TINA_evidence.zip.sha256").write_text(f"{digest(archive)}  TINA_evidence.zip\n")
    summary={"status":"prepared","created_utc":datetime.now(timezone.utc).isoformat(),"file_count":len(files),
        "checksummed_gib":total/2**30,"evaluations":evaluations,"predictions":predictions,
        "evidence_zip_mib":archive.stat().st_size/2**20,"evidence_files":len(archive_files),"weights_in_evidence_zip":False,
        "checksums":"migration/SHA256SUMS","manifest":"migration/file_manifest.json","next":"Transfer to destination, then run reproduction/verify_migration.py; no remote transfer has been performed."}
    write_json(OUT/"transfer_summary.json",summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

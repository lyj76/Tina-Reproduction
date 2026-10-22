"""Run the repository's official tasks using BF16 merged weights and LightEval."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="math_500,aime24,amc23")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prompt-seed", type=int, help="Fix randomized task construction before Pipeline initialization")
    parser.add_argument("--gpqa-csv", help="Explicit public GPQA CSV, verified against original HF Git blob")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {"model": args.model, "tasks": args.tasks, "engine_seed": args.seed,
                "sampling_seed": None, "temperature": 0.6, "top_p": 0.95,
                "prompt_seed": args.prompt_seed,
                "max_new_tokens": 32768, "max_model_length": 32768,
                "data_parallel_size": args.data_parallel_size, "dtype": "bfloat16", "use_chat_template": True,
                "gpu_memory_utilization": 0.5,
                "lighteval": importlib.metadata.version("lighteval"),
                "tasks_source": "scripts/eval/run_eval_custom_tasks.py",
                "tasks_source_sha256": hashlib.sha256(Path("scripts/eval/run_eval_custom_tasks.py").read_bytes()).hexdigest(),
                "max_samples": args.max_samples,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "notes": "ICLR 2026 Appendix B.1 command settings; BF16 merged weights, official tasks and LightEval 0.8.1."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tracker = EvaluationTracker(output_dir=str(out), save_details=True, push_to_hub=False,
                                push_to_tensorboard=False, public=False, hub_results_org=None)
    config = VLLMModelConfig(
        pretrained=str(Path(args.model).resolve()), dtype="bfloat16", seed=args.seed,
        data_parallel_size=args.data_parallel_size, use_chat_template=True, max_model_length=32768,
        gpu_memory_utilization=0.5,
        generation_parameters=GenerationParameters(max_new_tokens=32768, temperature=0.6, top_p=0.95))
    params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM, env_config=EnvConfig(), job_id=0,
        dataset_loading_processes=1, custom_tasks_directory="scripts/eval/run_eval_custom_tasks.py",
        override_batch_size=-1, num_fewshot_seeds=1, max_samples=args.max_samples,
        use_chat_template=True, system_prompt=None, load_responses_from_details_date_id=None)
    if args.gpqa_csv:
        import lighteval.tasks.lighteval_task as task_module
        from datasets import load_dataset
        csv_path = Path(args.gpqa_csv).resolve()
        raw = csv_path.read_bytes()
        oid = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        provenance = json.loads(Path("artifacts/core_suite/gpqa_public_provenance.json").read_text())
        assert oid == provenance["original_hf_file_metadata"]["oid"]
        original_download = task_module.download_dataset_worker
        def public_gpqa_download(dataset_path, dataset_config_name, *download_args, **download_kwargs):
            if dataset_path == "Idavidrein/gpqa" and dataset_config_name == "gpqa_diamond":
                return load_dataset("csv", data_files={"train": str(csv_path)})
            return original_download(dataset_path, dataset_config_name, *download_args, **download_kwargs)
        task_module.download_dataset_worker = public_gpqa_download
        manifest["gpqa_public_data"] = {"path": str(csv_path), "git_blob_oid": oid,
            "sha256": hashlib.sha256(raw).hexdigest(), "byte_identical_to_original_hf": True,
            "public_source": provenance["url"]}
    if args.prompt_seed is not None:
        # LightEval0.8.1 seeds Python only after task documents have been created.
        # The official GPQA prompt function randomizes the correct option position.
        random.seed(args.prompt_seed)
    pipeline = Pipeline(tasks=",".join(f"custom|{name}|0|0" for name in args.tasks.split(",")),
                        pipeline_parameters=params, evaluation_tracker=tracker,
                        model_config=config, metric_options={})
    input_specs = [{"task": request.task_name, "sample_index": request.sample_index,
                    "context": request.context, "num_samples": request.num_samples}
                   for requests in pipeline.requests.values() for request in requests]
    assert all(item["num_samples"] == 1 for item in input_specs)
    manifest["requests"] = len(input_specs)
    manifest["input_order_sha256"] = hashlib.sha256(json.dumps(input_specs, sort_keys=True).encode()).hexdigest()
    manifest["input_set_sha256"] = hashlib.sha256(json.dumps(sorted(input_specs, key=lambda x: (x["task"], str(x["sample_index"]))), sort_keys=True).encode()).hexdigest()
    merge_audit = Path(args.model) / "merge_audit.json"
    if merge_audit.exists():
        manifest["merge_audit"] = json.loads(merge_audit.read_text())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.prepare_only:
        assert args.data_parallel_size > 1 and pipeline.model.model is None
        requests = {str(key): [request.__dict__ for request in values]
                    for key, values in pipeline.requests.items()}
        (out / "prepared_requests.json").write_text(json.dumps(requests, indent=2, default=str))
        print({"prepared_requests": {key: len(values) for key, values in requests.items()},
               "GPU_model_allocated": False}, flush=True)
        raise SystemExit(0)
    pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
    pipeline.save_and_push_results()
    summary = {"manifest": manifest, "seconds": time.time() - started,
               "results": results["results"], "config_general": results["config_general"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)

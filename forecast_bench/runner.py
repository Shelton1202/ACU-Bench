"""Two-worker execution, stable batches, resumable outputs, and environment capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .io import (
    append_json,
    digest,
    load_predictions,
    read_json,
    save_json,
    valid_forecast,
)
from .protocol import PROTOCOL, SEED, prepare


def environment(config: dict) -> dict:
    record = {"python": platform.python_version(), "packages": {}}
    import torch

    record.update(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )
    if len(record["gpus"]) != 2 or any("T4" not in name for name in record["gpus"]):
        raise RuntimeError(f"This profile requires exactly two NVIDIA T4 GPUs: {record['gpus']}")
    if torch.__version__ != "2.11.0+cu128":
        raise RuntimeError(
            f"Expected torch 2.11.0+cu128, found {torch.__version__}; use the supplied environment"
        )
    expected = (
        {
            "transformers": "4.57.6",
            "accelerate": "1.11.0",
            "huggingface-hub": "0.36.2",
            "tokenizers": "0.22.2",
            "safetensors": "0.5.3",
        }
        if config["backend"] == "timebraid"
        else {
            "transformers": "4.51.3",
            "accelerate": "1.6.0",
            "bitsandbytes": "0.45.5",
            "huggingface-hub": "0.30.2",
            "lm-format-enforcer": "0.10.11",
        }
    )
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        record["packages"][name] = actual
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, found {actual}")
    return record


def initialize(root: Path, output: Path, slug: str, config: dict, tasks: list[dict]) -> Path:
    run_dir = output / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "protocol": PROTOCOL,
        "model": config,
        "seed": SEED,
        "dataset_sha256": digest((root / "dataset.jsonl").read_bytes()),
        "metadata_sha256": digest((root / "configs/samples.json").read_bytes()),
        "tasks_sha256": digest(tasks),
        "tasks": len(tasks),
        "code": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for directory in ("forecast_bench", "compat", "vendor")
            for p in sorted((root / directory).rglob("*.py"))
        },
    }
    manifest = run_dir / "manifest.json"
    if manifest.exists() and read_json(manifest) != signature:
        raise ValueError(f"Inputs or code changed; select a new output directory: {run_dir}")
    save_json(manifest, signature)
    return run_dir


def validate_coverage(run_dir: Path, tasks: list[dict], config: dict) -> dict:
    results = load_predictions(sorted(run_dir.glob("predictions_*.jsonl")))
    expected = {task["job_id"] for task in tasks}
    if set(results) != expected:
        raise ValueError(
            f"Incomplete task coverage for {config['name']}: {len(results)}/{len(expected)}"
        )
    if any(
        row.get("model") != config["name"] or row.get("protocol") != PROTOCOL
        for row in results.values()
    ):
        raise ValueError("Outputs contain a different model or prompt protocol")
    report = {
        "model": config["name"],
        "protocol": PROTOCOL,
        "valid_forecasts": len(results),
        "expected_forecasts": len(tasks),
        "complete": True,
    }
    save_json(run_dir / "completion.json", report)
    return report


def gpu_worker(root: Path, output: Path, slug: str, rank: int) -> None:
    import torch
    from transformers import set_seed

    from .adapters.language import LanguageModel
    from .adapters.native import Aurora, DoubleCast, TimeBraid

    config = read_json(root / "configs/models.json")[slug]
    _, _, all_tasks, _ = prepare(root)
    tasks = all_tasks[rank::2]
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    set_seed(SEED)
    run_dir = output / slug
    path = run_dir / f"predictions_{rank}.jsonl"
    done = load_predictions([path]) if path.exists() else {}
    if not set(done) <= {t["job_id"] for t in tasks}:
        raise ValueError("Wrong worker shard in resume file")
    classes = {
        "language": LanguageModel,
        "aurora": Aurora,
        "doublecast": DoubleCast,
        "timebraid": TimeBraid,
    }
    backend = config["backend"]
    model = (
        LanguageModel(config, rank)
        if backend == "language"
        else classes[backend](config, rank, run_dir, root)
    )
    batches = model.batches(tasks) if backend == "language" else [[task] for task in tasks]
    with path.open("a", encoding="utf-8") as stream:
        for batch in batches:
            batch_tasks = [item[1] for item in batch] if backend == "language" else batch
            if all(task["job_id"] in done for task in batch_tasks):
                continue
            records = model.predict(batch) if backend == "language" else [model.predict(batch[0])]
            for row in records:
                if not valid_forecast(row.get("prediction")):
                    raise ValueError(
                        f"Invalid forecast at {row['job_id']}; no imputation or repair"
                    )
                if row["job_id"] in done:
                    if row["prediction"] != done[row["job_id"]]["prediction"]:
                        raise ValueError("Recomputed partial batch changed; start a fresh run")
                    continue
                row.update(model=config["name"], protocol=PROTOCOL, rank=rank)
                append_json(stream, row)
                done[row["job_id"]] = row
            print(f"{slug} GPU {rank}: {len(done)}/{len(tasks)}", flush=True)
    save_json(
        run_dir / f"runtime_{rank}.json",
        {
            "complete": len(done) == len(tasks),
            "tasks": len(tasks),
            "gpu": torch.cuda.get_device_name(rank),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(rank),
        },
    )


def gpu_run(root: Path, output: Path, slug: str) -> dict:
    config = read_json(root / "configs/models.json")[slug]
    info = environment(config)
    _, _, tasks, _ = prepare(root)
    run_dir = initialize(root, output, slug, config, tasks)
    previous = run_dir / "environment.json"
    if previous.exists() and read_json(previous) != info:
        raise ValueError("Runtime environment changed; choose a new output directory")
    save_json(previous, info)
    if config["backend"] == "aurora":
        from huggingface_hub import snapshot_download

        snapshot_download(
            config["repo"],
            revision=config["revision"],
            local_dir=str(
                Path(os.environ.get("FORECAST_BENCH_MODEL_CACHE", run_dir)) / "aurora_pkg"
            ),
            allow_patterns=[
                "*.json",
                "*.py",
                "*.safetensors",
                "*.txt",
                "bert_config/*",
                "vit_config/*",
            ],
            max_workers=4,
        )
        (
            Path(os.environ.get("FORECAST_BENCH_MODEL_CACHE", run_dir)) / "aurora_pkg/__init__.py"
        ).touch()
    env = dict(os.environ, USE_TF="0", USE_FLAX="0")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    processes, logs = [], []
    try:
        for rank in (0, 1):
            log = (run_dir / f"worker_{rank}.log").open("w", encoding="utf-8")
            logs.append(log)
            command = [
                sys.executable,
                "-u",
                "-m",
                "forecast_bench",
                "--root",
                str(root),
                "_worker",
                "--model",
                slug,
                "--output",
                str(output),
                "--rank",
                str(rank),
            ]
            processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        while True:
            codes = [p.poll() for p in processes]
            if any(code is not None and code != 0 for code in codes):
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                codes = [p.wait() for p in processes]
                save_json(run_dir / "worker_exit_codes.json", codes)
                raise RuntimeError(f"GPU worker failure {codes}; inspect {run_dir}/worker_*.log")
            if all(code is not None for code in codes):
                break
            time.sleep(1)
        save_json(run_dir / "worker_exit_codes.json", codes)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
    return validate_coverage(run_dir, tasks, config)

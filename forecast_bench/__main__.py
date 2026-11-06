"""Run and evaluate forecasts with and without event text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from .io import read_json
from .protocol import PROTOCOL, prepare


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Source checkout containing the dataset and model configurations",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the configured models and inference backends")
    commands.add_parser("check", help="Verify benchmark task construction")
    descriptions = {
        "run": "Run one model on all numerical-only and text-conditioned tasks",
        "doctor": "Inspect the environment required by one model",
        "_worker": "Internal worker for a fixed GPU task shard",
    }
    for name in ("run", "doctor", "_worker"):
        sub = commands.add_parser(name, help=descriptions[name], description=descriptions[name])
        sub.add_argument("--model", required=True)
        sub.add_argument("--output", type=Path, default=Path("outputs"))
        if name == "_worker":
            sub.add_argument("--rank", type=int, choices=(0, 1), required=True)
    sub = commands.add_parser("evaluate", help="Score complete model runs and export result tables")
    sub.add_argument("--output", type=Path, default=Path("outputs"))
    sub.add_argument("--report", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    configs = read_json(root / "configs/models.json")
    if args.command == "list":
        for slug, config in configs.items():
            print(f"{slug:12} {config['backend']:12} {config['display']}")
        return
    if args.command == "check":
        rows, _, tasks, _ = prepare(root)
        result = {
            "protocol": PROTOCOL,
            "dataset_records": len(rows),
            "unique_inference_tasks": len(tasks),
        }
    elif args.command == "evaluate":
        from .metrics import evaluate

        result = evaluate(root, args.output.resolve(), args.report.resolve())
    else:
        from .runner import environment, gpu_run, gpu_worker

        config = configs[args.model]
        if args.command == "doctor":
            result = environment(config)
        elif args.command == "_worker":
            gpu_worker(root, args.output.resolve(), args.model, args.rank)
            return
        else:
            result = gpu_run(root, args.output.resolve(), args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

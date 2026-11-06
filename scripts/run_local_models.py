"""Run the six local models sequentially, each using both T4 GPUs."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3", "tablegpt2", "timeomni", "aurora", "doublecast", "timebraid"],
        choices=["qwen3", "tablegpt2", "timeomni", "aurora", "doublecast", "timebraid"],
    )
    args = parser.parse_args()
    for model in args.models:
        profile = "timebraid" if model == "timebraid" else "gpu"
        python = ROOT / f".venv-{profile}/bin/python"
        if not python.exists():
            raise RuntimeError(f"First run: python scripts/setup_environment.py {profile}")
        subprocess.run(
            [
                str(python),
                "-u",
                "-m",
                "forecast_bench",
                "--root",
                str(ROOT),
                "run",
                "--model",
                model,
                "--output",
                str(args.output.resolve()),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

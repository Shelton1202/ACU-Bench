"""Create one isolated GPU environment; model weights download at first run.

The commands are the same as those used for the reported runs: a Python 3.11
virtual environment populated by uv from the pinned lock file, with the CUDA
12.8 PyTorch index and uv's unsafe-best-match strategy used to compile the
locks, followed by an editable, dependency-free install of this repository.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("gpu", "timebraid"))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = (args.directory or ROOT / f".venv-{args.profile}").resolve()
    uv = [sys.executable, "-m", "uv"]
    if importlib.util.find_spec("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(uv + ["python", "install", "3.11"], check=True)
    subprocess.run(uv + ["venv", "--python", "3.11", str(directory)], check=True)
    python = str(directory / "bin/python")
    subprocess.run(
        uv
        + [
            "pip",
            "install",
            "--python",
            python,
            "-r",
            str(ROOT / "requirements" / f"{args.profile}.lock.txt"),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
            "--index-strategy",
            "unsafe-best-match",
        ],
        check=True,
    )
    subprocess.run(
        uv + ["pip", "install", "--python", python, "--no-deps", "-e", str(ROOT)], check=True
    )
    subprocess.run([python, "-u", "-m", "forecast_bench", "--root", str(ROOT), "check"], check=True)
    print(f"Environment ready: {python}")


if __name__ == "__main__":
    main()

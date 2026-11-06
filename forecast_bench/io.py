"""Strict, deterministic I/O for benchmark inputs and model outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    raw = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    )
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def append_json(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def valid_forecast(values: Any) -> bool:
    return (
        isinstance(values, list)
        and len(values) == 12
        and all(type(v) in (int, float) and math.isfinite(v) for v in values)
    )


def load_predictions(paths: list[Path]) -> dict[str, dict]:
    result = {}
    for path in paths:
        for row in read_jsonl(path):
            key = row["job_id"]
            if key in result:
                raise ValueError(f"Duplicate job in {path}: {key}")
            if not valid_forecast(row.get("prediction")):
                raise ValueError(f"Invalid twelve-step forecast in {path}: {key}")
            result[key] = row
    return result

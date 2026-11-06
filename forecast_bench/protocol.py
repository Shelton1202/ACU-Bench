"""Benchmark inputs, task identities, and the common forecasting prompt."""

from __future__ import annotations

import math
from pathlib import Path

from .io import digest, read_json, read_jsonl

SEED = 20261012
SYSTEM_PROMPT = "You are a helpful assistant."
PROTOCOL = "diagnosing-common-prompt-v1"
CONDITIONS = {
    "Q1": "Aligned",
    "Q2": "Aligned",
    "Q3": "Conflicting",
    "Q4": "Conflicting",
    "Q5": "Unspecified",
    "Q6": "Unspecified",
}
CELLS = {
    ("up", "up"): "Q1",
    ("down", "down"): "Q2",
    ("up", "down"): "Q3",
    ("down", "up"): "Q4",
    ("up", None): "Q5",
    ("down", None): "Q6",
}


def build_prompt(task: dict) -> str:
    unit = (
        "degrees Fahrenheit"
        if task["domain"] == "WeatherTemperature"
        else "the original observed units"
    )
    instruction = (
        f"Forecast the next 12 values of {task['target_variable']} ({task['domain']}, {unit}) "
        f"from the 96 historical values. Forecast origin: {task['forecast_origin']}. "
        "Identify the baseline pattern. If event text is provided, assess its incremental effect, "
        "allowing positive, negative, or zero impact. Avoid double-counting effects already present "
        "in history. Use only the supplied information. Return only a JSON object with forecast: "
        "an array of 12 numerical values."
    )
    history = "step,value\n" + "\n".join(
        f"{i + 1},{v}" for i, v in enumerate(task["past_time"]["value"])
    )
    return (
        instruction
        + "\nHistory:\n"
        + history
        + ("\nEvent/news context:\n" + task["text"] if task["text"] else "")
    )


def response_schema(bounded: bool = True) -> dict:
    array = {"type": "array", "items": {"type": "number"}}
    if bounded:
        array.update(minItems=12, maxItems=12)
    return {
        "type": "object",
        "properties": {"forecast": array},
        "required": ["forecast"],
        "additionalProperties": False,
    }


def prepare(root: Path) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Build only N/E tasks; future observations and q never enter model inputs."""
    rows = read_jsonl(root / "dataset.jsonl")
    metadata = read_json(root / "configs/samples.json")
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Expected a non-empty dataset with distinct sample IDs")
    if set(metadata) != {r["sample_id"] for r in rows}:
        raise ValueError("Evaluation metadata does not match dataset IDs")
    tasks, entries = {}, []
    for row in rows:
        for field, length in (("past_time", 96), ("future_time", 12)):
            value = row[field]
            if len(value["value"]) != length or len(value["timestamp"]) != length:
                raise ValueError(f"Incorrect window length: {row['sample_id']}")
            if not all(type(v) in (int, float) and math.isfinite(v) for v in value["value"]):
                raise ValueError(f"Non-finite observation: {row['sample_id']}")
        h = row["past_time"]["value"]
        delta = math.fsum(h[-12:]) - math.fsum(h[-24:-12])
        p = "up" if delta > 0 else "down" if delta < 0 else None
        if p != row["data_direction"] or CELLS.get((p, row["text_direction"])) != row["q"]:
            raise ValueError(f"History-based q mismatch: {row['sample_id']}")
        meta = metadata[row["sample_id"]]
        jobs = {}
        for condition in ("N", "E"):
            task = {
                "domain": meta["prompt_domain"],
                "original_domain": meta["original_domain"],
                "target_variable": row["target_variable"],
                "forecast_origin": meta["forecast_origin"],
                "past_time": row["past_time"],
                "text": row["text"] if condition == "E" else "",
            }
            job_id = digest(task)
            tasks[job_id] = {"job_id": job_id, **task}
            jobs[condition] = job_id
        entries.append(
            {"sample_id": row["sample_id"], "condition": CONDITIONS[row["q"]], "jobs": jobs}
        )
    return rows, metadata, [tasks[j] for j in sorted(tasks)], entries

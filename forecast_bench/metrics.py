"""Paper MASE and DA, with one result row per model and text-history relation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import load_predictions, read_json, save_json
from .protocol import PROTOCOL, prepare
from .runner import initialize, validate_coverage


def direction(values, baseline: float, domain: str) -> str:
    delta = float(np.mean(values)) - baseline
    if domain != "WeatherTemperature":
        delta /= abs(baseline) if baseline else 1.0
    threshold = {
        "WeatherTemperature": 3.6,
        "CommodityPrice": 0.02,
        "Currency": 0.005,
        "WikipediaPageviews": 0.10,
    }[domain]
    return "up" if delta > threshold else "down" if delta < -threshold else "stable"


def mase(rows: list[dict], key: str) -> float:
    groups = defaultdict(list)
    for row in rows:
        groups[row["variable"]].append(row)
    ratios = []
    for variable, group in groups.items():
        denominator = sum(row["naive"] for row in group)
        if denominator <= 0:
            raise ValueError(f"Zero seasonal-naive loss for variable {variable}")
        ratios.append(sum(row[key] for row in group) / denominator)
    if not ratios or not np.isfinite(ratios).all() or min(ratios) < 0:
        raise ValueError("Invalid MASE aggregation")
    return 0.0 if min(ratios) == 0 else float(np.exp(np.log(ratios).mean()))


def score_model(
    rows: list[dict], metadata: dict, entries: list[dict], predictions: dict
) -> list[dict]:
    samples = {row["sample_id"]: row for row in rows}
    scored = []
    for entry in entries:
        sample = samples[entry["sample_id"]]
        meta = metadata[entry["sample_id"]]
        history = np.asarray(sample["past_time"]["value"], dtype=float)
        future = np.asarray(sample["future_time"]["value"], dtype=float)
        period = meta["seasonality"]
        scale = float(np.abs(history[period:] - history[:-period]).mean())
        if scale <= 0:
            raise ValueError(f"Zero historical scale for {entry['sample_id']}")
        baseline = float(history[-12:].mean())
        truth = direction(future, baseline, meta["original_domain"])
        record = {
            "sample_id": entry["sample_id"],
            "condition": entry["condition"],
            "variable": meta["variable"],
            "naive": float(np.abs(np.resize(history[-period:], 12) - future).mean() / scale),
        }
        for condition in ("N", "E"):
            forecast = np.asarray(predictions[entry["jobs"][condition]]["prediction"], dtype=float)
            record["L_" + condition] = float(np.abs(forecast - future).mean() / scale)
            record["DA_" + condition] = int(
                direction(forecast, baseline, meta["original_domain"]) == truth
            )
        scored.append(record)
    return scored


def summarize(scored: list[dict], model: str) -> list[dict]:
    result = []
    for condition in ("Aligned", "Conflicting", "Unspecified"):
        group = [row for row in scored if row["condition"] == condition]
        result.append(
            {
                "Model": model,
                "Relation": condition,
                "windows": len(group),
                "L_N": mase(group, "L_N"),
                "L_E": mase(group, "L_E"),
                "DA_N": float(np.mean([r["DA_N"] for r in group]) * 100),
                "DA_E": float(np.mean([r["DA_E"] for r in group]) * 100),
            }
        )
    return result


def evaluate(root: Path, output: Path, report: Path) -> dict:
    models = read_json(root / "configs/models.json")
    rows, metadata, tasks, entries = prepare(root)
    results = []
    for slug, config in models.items():
        run_dir = output / slug
        if not (run_dir / "manifest.json").exists():
            raise ValueError(f"Missing completed run: {slug}")
        initialize(root, output, slug, config, tasks)  # verify protocol and input identity
        validate_coverage(run_dir, tasks, config)
        predictions = load_predictions(sorted(run_dir.glob("predictions_*.jsonl")))
        scored = score_model(rows, metadata, entries, predictions)
        save_json(report / "per_sample" / f"{slug}.json", scored)
        results.extend(summarize(scored, config["display"]))
    save_json(report / "scores.json", results)
    write_tables(results, report)
    summary = {
        "protocol": PROTOCOL,
        "models": len(models),
        "unique_tasks_per_model": len(tasks),
        "scored_text_window_pairs_per_model": len(rows),
        "windows_per_condition": {r["Relation"]: r["windows"] for r in results[:3]},
    }
    save_json(report / "summary.json", summary)
    return summary


def write_tables(rows: list[dict], report: Path) -> None:
    report.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(r["Model"] for r in rows))
    lookup = {(r["Model"], r["Relation"]): r for r in rows}
    conditions = ("Aligned", "Conflicting", "Unspecified")
    counts = {}
    for row in rows:
        if counts.setdefault(row["Relation"], row["windows"]) != row["windows"]:
            raise ValueError("Models were scored on different pairs")
    caption = (
        "Forecasting with and without text. Bold marks the better score, or both scores "
        "in a tie; the last row counts models improving with text, excluding ties."
    )
    headers = ["Model"] + [
        f"{c} {m} {t}"
        for c in conditions
        for m in ("MASE", "DA (%)")
        for t in ("w/o text", "w/ text")
    ]
    markdown = [
        caption,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 13) + " |",
    ]
    tex_rows = []
    for name in names:
        md_cells, tex_cells = [name], [name]
        for condition in conditions:
            row = lookup[name, condition]
            for metric in ("L", "DA"):
                a, b = row[metric + "_N"], row[metric + "_E"]
                better = (a <= b, b <= a) if metric == "L" else (a >= b, b >= a)
                for value, bold in zip((a, b), better, strict=True):
                    text = f"{value:.4f}" if metric == "L" else f"{value:.2f}"
                    md_cells.append(f"**{text}**" if bold else text)
                    tex_cells.append(r"\textbf{" + text + "}" if bold else text)
        markdown.append("| " + " | ".join(md_cells) + " |")
        tex_rows.append(" & ".join(tex_cells) + r" \\")
    improved = []
    for condition in conditions:
        for metric in ("L", "DA"):
            wins = sum(
                lookup[name, condition][metric + "_E"] < lookup[name, condition][metric + "_N"]
                if metric == "L"
                else lookup[name, condition][metric + "_E"] > lookup[name, condition][metric + "_N"]
                for name in names
            )
            improved.append(f"{wins}/{len(names)}")
    markdown.append(
        "| Improved models | "
        + " | ".join(cell for value in improved for cell in (value, ""))
        + " |"
    )
    tex_rows += [
        r"\midrule",
        "Improved models & "
        + " & ".join(r"\multicolumn{2}{c}{" + value + "}" for value in improved)
        + r" \\",
    ]
    tex = (
        r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{12}{c}@{}}
\toprule
\multirow{3}{*}{Model} & """
        + " & ".join(r"\multicolumn{4}{c}{" + c + f" ($n={counts[c]}$)" + "}" for c in conditions)
        + r""" \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
 & \multicolumn{2}{c}{MASE $\downarrow$} & \multicolumn{2}{c}{DA(\%) $\uparrow$} & \multicolumn{2}{c}{MASE $\downarrow$} & \multicolumn{2}{c}{DA(\%) $\uparrow$} & \multicolumn{2}{c}{MASE $\downarrow$} & \multicolumn{2}{c}{DA(\%) $\uparrow$} \\
 & \shortstack{w/o\\text} & \shortstack{w/\\text} & \shortstack{w/o\\text} & \shortstack{w/\\text} & \shortstack{w/o\\text} & \shortstack{w/\\text} & \shortstack{w/o\\text} & \shortstack{w/\\text} & \shortstack{w/o\\text} & \shortstack{w/\\text} & \shortstack{w/o\\text} & \shortstack{w/\\text} \\
\midrule
"""
        + "\n".join(tex_rows)
        + "\n"
        + r"""\bottomrule
\end{tabular*}
\caption{"""
        + caption
        + r"""}
\label{tab:overall_window_performance}
\end{table*}
"""
    )
    (report / "benchmark_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (report / "benchmark_results.tex").write_text(tex, encoding="utf-8")

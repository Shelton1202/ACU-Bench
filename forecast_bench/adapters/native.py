"""Native Aurora, DoubleCast, and TimeBraid inference paths for two T4 workers."""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path

from forecast_bench.io import digest


class Aurora:
    def __init__(self, config: dict, rank: int, run_dir: Path, root: Path):
        model_cache = Path(os.environ.get("FORECAST_BENCH_MODEL_CACHE", run_dir))
        sys.path.insert(0, str(model_cache))
        from aurora_pkg.modeling_aurora import AuroraForPrediction

        self.device = f"cuda:{rank}"
        self.model = (
            AuroraForPrediction.from_pretrained(model_cache / "aurora_pkg").to(self.device).eval()
        )
        self.model.requires_grad_(False)
        self.tokenizer = self.model.tokenizer
        self.run_dir = run_dir

    def predict(self, task: dict) -> dict:
        import numpy as np
        import torch
        from transformers import set_seed

        history = torch.tensor(
            [task["past_time"]["value"]], dtype=torch.float32, device=self.device
        )
        kwargs = {}
        token_count = 0
        if task["text"]:
            token_count = len(self.tokenizer.encode(task["text"], add_special_tokens=True))
            encoded = self.tokenizer(
                [task["text"]],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            kwargs = {
                "text_input_ids": encoded["input_ids"].to(self.device),
                "text_attention_mask": encoded["attention_mask"].to(self.device),
                "text_token_type_ids": encoded.get(
                    "token_type_ids", torch.zeros_like(encoded["input_ids"])
                ).to(self.device),
            }
        seed = int(digest(task["past_time"])[:8], 16)
        set_seed(seed)
        with torch.inference_mode():
            draws = self.model.generate(
                inputs=history,
                max_output_length=12,
                num_samples=20,
                inference_token_len=12,
                **kwargs,
            )
        if tuple(draws.shape) != (1, 20, 12):
            raise ValueError(f"Unexpected Aurora output shape: {tuple(draws.shape)}")
        path = self.run_dir / "draws" / f"{task['job_id']}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, draws=draws.cpu().numpy())
        return {
            "job_id": task["job_id"],
            "prediction": draws.mean(1)[0].cpu().tolist(),
            "seed": seed,
            "ensemble_samples": 20,
            "text_truncated": token_count > 512,
            "text_token_limit": 512,
        }


def historical_frequency(task: dict) -> str:
    domain = task["original_domain"]
    times = [
        datetime.fromisoformat(t.replace("Z", "+00:00")) for t in task["past_time"]["timestamp"]
    ]
    steps = [int((b - a).total_seconds()) for a, b in zip(times, times[1:])]
    if not steps or min(steps) <= 0:
        raise ValueError("Non-increasing historical timestamps")
    if domain in ("CommodityPrice", "Currency"):
        recent = steps[-24:]
        if len(recent) >= 12 and sum(step == 7 * 86400 for step in recent) / len(recent) >= 0.8:
            weekdays = [timestamp.weekday() for timestamp in times[-25:]]
            day, count = Counter(weekdays).most_common(1)[0]
            if count / len(weekdays) < 0.8:
                raise ValueError("Weekly history does not have a consistent weekday")
            return "W-" + ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")[day]
        return "B"
    step, count = Counter(steps).most_common(1)[0]
    expected = {"WeatherTemperature": 3600, "WikipediaPageviews": 86400}[domain]
    if step != expected or count / len(steps) < 0.8:
        raise ValueError("Unsupported historical frequency")
    return "h" if step == 3600 else "D"


class DoubleCast:
    def __init__(self, config: dict, rank: int, run_dir: Path, root: Path):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, BitsAndBytesConfig

        sys.path.insert(0, str(root / "vendor/doublecast_source"))
        import doublecast.models.doublecast as dc
        from chronos import ChronosConfig
        from doublecast.models.doublecast import (
            DoubleCastConfig,
            DoubleCastModel,
            DoubleCastPipeline,
        )

        patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]
        text_path = snapshot_download(
            "Qwen/Qwen3-14B",
            revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
            allow_patterns=patterns,
            max_workers=2,
        )
        chronos_path = snapshot_download(
            "amazon/chronos-t5-large",
            revision="0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a",
            allow_patterns=patterns,
            max_workers=2,
        )
        checkpoint = snapshot_download(
            config["repo"], revision=config["revision"], allow_patterns=patterns, max_workers=2
        )
        base_loader = dc.AutoModel.from_pretrained

        def quantized_loader(path, *args, **kwargs):
            kwargs.update(
                device_map={"": rank},
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
            )
            return base_loader(path, *args, **kwargs)

        dc.AutoModel.from_pretrained = quantized_loader
        model_config = DoubleCastConfig.from_pretrained(checkpoint)
        model_config.text_encoder_path = text_path
        model_config.chronos_path = chronos_path
        try:
            with torch.device("cpu"):
                model = DoubleCastModel(model_config)
        finally:
            dc.AutoModel.from_pretrained = base_loader
        model.model.to(device=f"cuda:{rank}", dtype=torch.float16)
        weights = DoubleCastModel._load_state_dict_from_checkpoint(Path(checkpoint), "cpu")
        expected = {
            k
            for k in model.state_dict()
            if k.startswith("model.decoder.block.") and (".layer.2." in k or ".layer.3." in k)
        }
        if not expected or not expected.issubset(weights):
            raise ValueError("Incomplete DoubleCast delta checkpoint")
        _, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected DoubleCast weights: {unexpected}")
        model._tie_weights()
        model.eval().requires_grad_(False)
        self.pipeline = DoubleCastPipeline(
            tokenizer=ChronosConfig(**model_config.chronos_config).create_tokenizer(),
            text_tokenizer=AutoTokenizer.from_pretrained(text_path),
            model=model,
        )
        self.run_dir = run_dir

    def predict(self, task: dict) -> dict:
        import numpy as np
        import pandas as pd
        import torch
        from doublecast.data.prompts import prompt_structured
        from transformers import set_seed

        frequency = historical_frequency(task)
        past = pd.DatetimeIndex(pd.to_datetime(task["past_time"]["timestamp"]))
        future = pd.date_range(past[-1], periods=13, freq=frequency)[1:]
        formatter = partial(
            prompt_structured, past_timestamps=past, future_timestamps=future, freq=frequency
        )
        seed = int(digest(task["past_time"])[:8], 16)
        set_seed(seed)
        with torch.inference_mode():
            draws = self.pipeline.predict(
                context=torch.tensor(task["past_time"]["value"], dtype=torch.float32),
                text_context=task["text"],
                prediction_length=12,
                num_samples=20,
                limit_prediction_length=False,
                format_prompt=formatter,
            )
        if tuple(draws.shape) != (1, 20, 12):
            raise ValueError(f"Unexpected DoubleCast output shape: {tuple(draws.shape)}")
        path = self.run_dir / "draws" / f"{task['job_id']}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, draws=draws.numpy())
        return {
            "job_id": task["job_id"],
            "prediction": draws.mean(1)[0].tolist(),
            "seed": seed,
            "ensemble_samples": 20,
        }


def timebraid_prompt(task: dict) -> str:
    """Preserve TimeBraid's original native prompt."""
    unit = (
        "degrees Fahrenheit"
        if task["domain"] == "WeatherTemperature"
        else "the original observed units"
    )
    instruction = (
        f"Forecast the next 12 observations of {task['target_variable']} in {unit}. "
        f"Domain: {task['domain']}. Forecast origin: {task['forecast_origin']}. "
        "Use the observed time series and only the supplied context. "
        "Return a numerical forecast for the target series."
    )
    if task["text"]:
        instruction += "\nEvent/news context:\n" + task["text"]
    return instruction


class TimeBraid:
    def __init__(self, config: dict, rank: int, run_dir: Path, root: Path):
        import torch

        sys.path.insert(0, str(root / "compat"))
        sys.path.insert(0, str(root / "vendor"))
        import t4_attention_compat as compat
        from transformers import AutoModelForCausalLM, AutoProcessor

        compat.install()
        from timebraid import TimeBraidProcessor  # noqa: F401 -- register AutoClasses

        compat.patch_timebraid_dtype_reconciliation()
        self.processor = AutoProcessor.from_pretrained(
            config["repo"],
            revision=config["revision"],
            fix_mistral_regex=False,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["repo"],
            revision=config["revision"],
            dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": rank},
            trust_remote_code=False,
            use_safetensors=True,
        ).eval()
        self.model.requires_grad_(False)

    def predict(self, task: dict) -> dict:
        import torch

        prompt = timebraid_prompt(task)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            timeseries=[task["past_time"]["value"]],
            horizon=12,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            raw = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False, num_beams=1, num_return_sequences=1
            )
        result = self.processor.post_process_generation(raw, model_inputs=inputs)
        values = result["timeseries"]["values"] if result["timeseries"] else None
        return {
            "job_id": task["job_id"],
            "prediction": list(values) if values is not None else None,
            "prompt_sha256": digest(prompt.encode()),
            "attention_backend": "torch_sdpa_t4",
        }

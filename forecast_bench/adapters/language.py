"""The three local instruction models use the common prompt and JSON-object output."""

from __future__ import annotations

import json

from forecast_bench.io import digest, valid_forecast
from forecast_bench.protocol import SYSTEM_PROMPT, build_prompt, response_schema


class LanguageModel:
    def __init__(self, config: dict, rank: int):
        import torch
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["repo"], revision=config["revision"], padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config["repo"],
            revision=config["revision"],
            torch_dtype=torch.float16,
            device_map={"": rank},
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        ).eval()
        self.model.requires_grad_(False)
        self.token_data = build_token_enforcer_tokenizer_data(self.tokenizer)

    def batches(self, tasks: list[dict]):
        jobs = []
        for task in tasks:
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(task)},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **({"enable_thinking": False} if self.config["name"] == "Qwen3-8B" else {}),
            )
            jobs.append((len(self.tokenizer.encode(text)), task, text))
        jobs.sort(key=lambda item: (item[0], item[1]["job_id"]))
        batch = []
        for job in jobs:
            if batch and (len(batch) == 8 or job[0] * (len(batch) + 1) > 16000):
                yield batch
                batch = []
            batch.append(job)
        if batch:
            yield batch

    def predict(self, batch: list) -> list[dict]:
        import torch
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        encoded = self.tokenizer([item[2] for item in batch], padding=True, return_tensors="pt").to(
            self.model.device
        )
        if encoded.input_ids.shape[1] + 2048 > self.model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds the model context; truncation is disabled")
        constraint = build_transformers_prefix_allowed_tokens_fn(
            self.token_data, JsonSchemaParser(response_schema())
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=2048,
                do_sample=False,
                prefix_allowed_tokens_fn=constraint,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        decoded = self.tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        records = []
        for (_, task, chat), raw in zip(batch, decoded, strict=True):
            values = json.loads(raw.strip())["forecast"]
            if not valid_forecast(values):
                raise ValueError(f"Invalid output for {task['job_id']}: {raw!r}")
            records.append(
                {
                    "job_id": task["job_id"],
                    "prediction": values,
                    "raw_output": raw,
                    "prompt_sha256": digest(build_prompt(task).encode()),
                    "chat_sha256": digest(chat.encode()),
                }
            )
        return records

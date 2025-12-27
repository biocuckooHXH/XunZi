# src/model/mistral_wrapper.py
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class MistralForBiomedicalCPT(nn.Module):
    def __init__(
        self,
        model_name: str,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        torch_dtype: str = "float16",  # "float16" or "bfloat16"
    ):
        super().__init__()

        dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Optional[Dict]):
        from peft import LoraConfig, TaskType, get_peft_model

        if lora_config is None:
            lora_config = dict(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )

        peft_cfg = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, peft_cfg)
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save_pretrained(self, save_dir: str):
        self.model.save_pretrained(save_dir)

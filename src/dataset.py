# src/data/dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BiomedicalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 2048,
    ):
        self.data_path = Path(data_path)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data = self._load_jsonl(self.data_path)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict]:
        items = []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                items = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        text = self.data[idx]["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # pad in collator
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # CLM labels = input_ids (padding mask will be handled in collator)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class DataCollatorForCausalLM:
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # pad to max length in batch
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100  # ignore padding tokens
        batch["labels"] = labels
        return batch

# src/data/preprocessor.py
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    total: int = 0
    kept: int = 0
    too_short: int = 0
    too_long: int = 0
    error: int = 0


class BiomedicalTextPreprocessor:
    """
    Preprocess biomedical literature / UniProt text for:
      - CPT (continued pretraining): pure text
      - IT  (instruction tuning): prompt template + text
    """

    def __init__(
        self,
        tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        max_length: int = 2048,
        min_length: int = 128,
        task: str = "cpt",  # "cpt" or "it"
        seed: int = 42,
    ):
        assert task in ["cpt", "it"], "task must be one of: ['cpt', 'it']"
        self.max_length = int(max_length)
        self.min_length = int(min_length)
        self.task = task
        self.rng = random.Random(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def clean_text(text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # normalize quotes
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        return text.strip()

    def format_record(self, record: Dict) -> str:
        """
        Expected record fields (flexible):
          - pmid / id
          - title
          - abstract / text / content
          - genes (optional list[str])
          - diseases (optional list[str])
          - associations (optional list[dict])
        """
        pmid = str(record.get("pmid") or record.get("id") or "")
        title = self.clean_text(record.get("title", ""))
        abstract = self.clean_text(record.get("abstract") or record.get("text") or record.get("content") or "")

        genes = record.get("genes") or []
        diseases = record.get("diseases") or []
        associations = record.get("associations") or []

        # For CPT: keep it simple & stable
        parts = []
        if pmid:
            parts.append(f"[PMID] {pmid}")
        if title:
            parts.append(f"[TITLE] {title}")
        if abstract:
            parts.append(f"[TEXT] {abstract}")

        if genes:
            parts.append(f"[GENES] {', '.join(map(str, genes))}")
        if diseases:
            parts.append(f"[DISEASES] {', '.join(map(str, diseases))}")

        if associations:
            parts.append("[ASSOCIATIONS]")
            for assoc in associations:
                g = assoc.get("gene", "")
                d = assoc.get("disease", "")
                rel = assoc.get("relation", "associated_with")
                conf = assoc.get("confidence", "medium")
                parts.append(f"- {g} {rel} {d} (confidence: {conf})")

        text = "\n".join(parts).strip()
        return text

    def wrap_for_task(self, text: str) -> str:
        if self.task == "cpt":
            # pure text for continued pretraining
            return text

        # instruction tuning mode (optional)
        templates = [
            "<s>[INST] Extract biomedical entities and relationships from the following text:\n{t}\n[/INST]",
            "<s>[INST] Summarize key gene-disease associations in the following passage:\n{t}\n[/INST]",
            "<s>[INST] Analyze the biomedical text and identify mechanistic links between genes and diseases:\n{t}\n[/INST]",
        ]
        return self.rng.choice(templates).format(t=text)

    def smart_truncate(self, text: str, max_tokens: int) -> str:
        """
        Keep priority sections first, then append remaining lines until token limit.
        """
        lines = text.split("\n")
        priority_tags = ("[GENES]", "[DISEASES]", "[ASSOCIATIONS]", "[TITLE]", "[TEXT]", "[PMID]")

        priority = [ln for ln in lines if any(tag in ln for tag in priority_tags)]
        others = [ln for ln in lines if ln not in priority]

        result = "\n".join(priority).strip()
        for ln in others:
            candidate = (result + "\n" + ln).strip()
            if len(self.tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens:
                result = candidate
            else:
                break
        return result

    def process_records(self, records: List[Dict]) -> List[Dict]:
        stats = PreprocessStats(total=len(records))
        out: List[Dict] = []

        for rec in tqdm(records, desc="Preprocessing"):
            try:
                text = self.format_record(rec)
                text = self.wrap_for_task(text)

                toks = self.tokenizer.encode(text, add_special_tokens=False)
                if len(toks) < self.min_length:
                    stats.too_short += 1
                    continue

                if len(toks) > self.max_length:
                    stats.too_long += 1
                    text = self.smart_truncate(text, self.max_length)
                    toks = self.tokenizer.encode(text, add_special_tokens=False)

                out.append(
                    {
                        "text": text,
                        "meta": {
                            "pmid": str(rec.get("pmid") or rec.get("id") or ""),
                            "token_length": len(toks),
                        },
                    }
                )
                stats.kept += 1
            except Exception as e:
                stats.error += 1
                logger.warning(f"Failed record: {e}")
                continue

        logger.info(
            f"Done. total={stats.total}, kept={stats.kept}, "
            f"too_short={stats.too_short}, too_long={stats.too_long}, error={stats.error}"
        )
        return out

# scripts/train.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import BiomedicalDataset, DataCollatorForCausalLM
from src.model.mistral_wrapper import MistralForBiomedicalCPT
from src.training.trainer import BiomedicalTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(cfg_path: str, use_wandb: bool):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    torch.manual_seed(cfg.get("seed", 42))

    model_name = cfg["model"]["name"]
    max_length = cfg["data"]["max_length"]

    train_ds = BiomedicalDataset(cfg["data"]["train_path"], tokenizer_name=model_name, max_length=max_length)
    val_path = cfg["data"].get("val_path")
    val_ds = BiomedicalDataset(val_path, tokenizer_name=model_name, max_length=max_length) if val_path else None

    collator = DataCollatorForCausalLM(tokenizer_name=model_name)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collator,
    )

    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=True,
            collate_fn=collator,
        )

    model = MistralForBiomedicalCPT(
        model_name=model_name,
        use_lora=cfg["model"].get("use_lora", True),
        lora_config=cfg["model"].get("lora_config"),
        torch_dtype=cfg["model"].get("torch_dtype", "float16"),
    )

    trainer = BiomedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg["training"],
        use_wandb=use_wandb,
    )

    logger.info("Start training...")
    trainer.train()
    logger.info("Training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()
    main(args.config, args.use_wandb)

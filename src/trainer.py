# src/training/trainer.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class BiomedicalTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict,
        use_wandb: bool = False,
    ):
        self.config = config
        self.use_wandb = use_wandb

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            mixed_precision=config.get("mixed_precision", "fp16"),
        )

        self.optimizer = AdamW(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            eps=config.get("eps", 1e-8),
        )

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            model, self.optimizer, train_loader, val_loader
        )

        # scheduler
        num_update_steps_per_epoch = max(1, len(self.train_loader) // self.accelerator.gradient_accumulation_steps)
        self.num_training_steps = num_update_steps_per_epoch * config["num_epochs"]
        self.num_warmup_steps = int(self.num_training_steps * config.get("warmup_ratio", 0.1))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        self.scheduler = self.accelerator.prepare(self.scheduler)

        # wandb optional (only main process)
        self.wandb = None
        if use_wandb and self.accelerator.is_main_process:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=config.get("wandb_project", "biomedical-cpt"), config=config, name=config.get("run_name"))

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        losses = []

        for batch in tqdm(self.val_loader, desc="Valid", disable=not self.accelerator.is_local_main_process):
            out = self.model(**batch)
            loss = out.loss
            losses.append(self.accelerator.gather(loss.detach()).float().cpu())

        loss = torch.cat(losses).mean().item()
        ppl = float(torch.exp(torch.tensor(loss)))
        metrics = {"val/loss": loss, "val/ppl": ppl, "epoch": epoch}

        if self.wandb and self.accelerator.is_main_process:
            self.wandb.log(metrics)

        self.model.train()
        return metrics

    def save_checkpoint(self, name: str, extra: Optional[Dict] = None):
        if not self.accelerator.is_main_process:
            return
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(str(save_path))

        state = {
            "config": self.config,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if extra:
            state.update(extra)

        torch.save(state, save_path / "trainer_state.pt")
        with (save_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Saved checkpoint: {save_path}")

    def train(self):
        best_val_loss = float("inf")

        global_step = 0
        self.model.train()

        for epoch in range(self.config["num_epochs"]):
            progress = tqdm(self.train_loader, desc=f"Train epoch {epoch}", disable=not self.accelerator.is_local_main_process)

            running = 0.0
            for step, batch in enumerate(progress):
                with self.accelerator.accumulate(self.model):
                    out = self.model(**batch)
                    loss = out.loss
                    running += loss.detach().float()

                    self.accelerator.backward(loss)

                    if self.config.get("max_grad_norm", 1.0) > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1
                if step % self.config.get("log_every", 10) == 0:
                    avg_loss = (running / (step + 1)).item()
                    lr = self.scheduler.get_last_lr()[0]
                    progress.set_postfix(loss=avg_loss, lr=lr)

                    if self.wandb and self.accelerator.is_main_process:
                        self.wandb.log({"train/loss": avg_loss, "train/lr": lr, "step": global_step, "epoch": epoch})

            # validate
            metrics = self.evaluate(epoch)
            val_loss = metrics.get("val/loss", None)

            # save checkpoint every epoch (lightweight for LoRA)
            if (epoch + 1) % self.config.get("save_every", 1) == 0:
                self.save_checkpoint(f"checkpoint-epoch-{epoch}")

            # save best
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model", extra={"best_val_loss": best_val_loss, "best_epoch": epoch})

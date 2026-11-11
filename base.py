"""Shared Lightning training behavior for single- and multi-label classification."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Accuracy, AveragePrecision, F1Score, MetricCollection, Precision, Recall


class BaseModel(pl.LightningModule):
    """Single-device training with exact epoch metrics from detached CPU predictions."""

    def __init__(self, args: Namespace, datamodule: pl.LightningDataModule, network: nn.Module):
        super().__init__()
        # Keep checkpoint metadata compatible with PyTorch's safe weights-only loader.
        self.save_hyperparameters(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
        )
        self.network = network
        self.task = datamodule.task
        self.num_classes = datamodule.num_classes
        self.class_names = tuple(datamodule.class_names)
        self.criterion = self.init_criterion()
        self.metrics = self.init_metrics()
        self.training_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

    @property
    def monitor(self) -> str:
        metric = "accuracy_macro" if self.task == "multiclass" else "average_precision_macro"
        return f"val/{metric}"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def init_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss() if self.task == "multiclass" else nn.BCEWithLogitsLoss()

    def init_metrics(self) -> MetricCollection:
        definitions = {"f1": F1Score}
        if self.task == "multiclass":
            definitions.update(accuracy=Accuracy, precision=Precision, recall=Recall)
        else:
            definitions["average_precision"] = AveragePrecision
        kwargs = dict(task=self.task, num_classes=self.num_classes, num_labels=self.num_classes)
        return MetricCollection(
            {
                f"{name}_{average or 'per_class'}": metric(average=average, **kwargs)
                for name, metric in definitions.items()
                for average in ("micro", "macro", None)
            },
            compute_groups=False,
        )

    def _step(self, batch, outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        targets = labels.long() if self.task == "multiclass" else labels.float()
        loss = self.criterion(logits, targets)
        probabilities = logits.softmax(dim=1) if self.task == "multiclass" else logits.sigmoid()
        outputs.append(
            {
                "labels": labels.detach().long().cpu(),
                "probabilities": probabilities.detach().float().cpu(),
                "loss": loss.detach().float().cpu(),
            }
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, self.training_step_outputs)

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, self.validation_step_outputs)

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, self.test_step_outputs)

    def _finish_epoch(self, phase: str, outputs: list[dict[str, torch.Tensor]]) -> None:
        if not outputs:
            return
        # Each hook computes and resets immediately, so all phases share one collection.
        metrics = self.metrics.cpu()
        try:
            labels = torch.cat([output["labels"] for output in outputs])
            probabilities = torch.cat([output["probabilities"] for output in outputs])
            metrics.update(probabilities, labels)
            logged = {}
            for name, value in metrics.compute().items():
                if name.endswith("_per_class"):
                    metric_name = name.removesuffix("_per_class")
                    for class_name, class_value in zip(self.class_names, value, strict=True):
                        logged[f"{phase}/{metric_name}/{class_name}"] = class_value.to(self.device)
                else:
                    logged[f"{phase}/{name}"] = value.to(self.device)
            # Weight partial last batches by sample count, not by batch count.
            loss = sum(output["loss"] * len(output["labels"]) for output in outputs) / len(labels)
            logged[f"{phase}/loss"] = loss.to(self.device)
            self.log_dict(logged, on_step=False, on_epoch=True, batch_size=len(labels))
        finally:
            metrics.reset()
            outputs.clear()

    def on_train_epoch_end(self) -> None:
        self._finish_epoch("train", self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._finish_epoch("val", self.validation_step_outputs)

    def on_test_epoch_end(self) -> None:
        self._finish_epoch("test", self.test_step_outputs)

    def setup(self, stage: str) -> None:
        if self.trainer.world_size != 1:
            raise RuntimeError("BaseModel currently requires devices=1 for exact epoch metrics.")

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 0.001)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=self.hparams.get("weight_decay", 0.01)
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        pct_start = self.hparams.get("pct_start", 0.3)
        # OneCycleLR divides by zero when its warmup occupies exactly one step.
        if abs(pct_start * total_steps - 1) < 1e-12:
            pct_start += 1e-6
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

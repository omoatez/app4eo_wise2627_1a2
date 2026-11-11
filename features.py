"""Capture the input to the final linear layer on unaugmented training samples."""

from pathlib import Path

import lightning as L
import numpy as np
import torch


class FeatureExtractor(L.Callback):
    """Write float16 features at one-based epoch numbers, with labels in sample order."""

    def __init__(self, output_dir: str | Path, epochs: tuple[int, ...] = (5, 10)):
        self.output_dir = Path(output_dir)
        self.epochs = set(epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch not in self.epochs:
            return
        if trainer.world_size != 1:
            raise ValueError("Feature extraction currently requires a single device.")

        features, labels = [], []

        def capture(_module, inputs, _output):
            features.append(inputs[0].detach().to(device="cpu", dtype=torch.float16).numpy())

        classifier = pl_module.network.get_classifier()
        handle = classifier.register_forward_hook(capture)
        was_training = pl_module.training
        try:
            pl_module.eval()
            with torch.inference_mode():
                for images, targets in trainer.datamodule.feature_dataloader():
                    pl_module(images.to(pl_module.device))
                    labels.append(targets.cpu().numpy())
        finally:
            handle.remove()
            pl_module.train(was_training)

        if not features:
            raise ValueError("Cannot extract features from an empty training set.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        targets = np.concatenate(labels)
        np.savez_compressed(
            self.output_dir / f"epoch-{epoch:02d}.npz",
            features=np.concatenate(features),
            labels=targets,
            sample_indices=np.arange(len(targets)),
            class_names=np.asarray(trainer.datamodule.class_names),
            epoch=epoch,
        )

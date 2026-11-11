"""Entry point: load data, build a network, train, then test its best checkpoint."""

import argparse
import json
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from base import BaseModel
from data import Caltech101DataModule, EuroSATDataModule, TinyBENDataModule
from data.transform import AUGMENTATIONS, build_transforms, fit_rgb_statistics, fit_statistics
from features import FeatureExtractor
from models import get_network

DATASETS = {
    "eurosat": (EuroSATDataModule, "EuroSAT"),
    "tinyben": (TinyBENDataModule, "tinyBEN"),
    "caltech101": (Caltech101DataModule, "caltech101"),
}


def dataset_path(folder):
    """Use the GPU runner's shared data mount, or the local project datasets."""
    root = Path(os.environ.get("DATA_DIR", "/data")) / "untracked-files"
    if not root.is_dir() and "DATA_DIR" not in os.environ:
        root = Path(__file__).resolve().parent / "untracked-files"
    return root / folder


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default="eurosat")
    p.add_argument("--dataset-path", type=Path)
    p.add_argument("--arch-name", choices=("ResNet-6", "ResNet18", "ViT-Tiny"), default="ResNet-6")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pct-start", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", choices=AUGMENTATIONS, default="flip")
    p.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu", "mps"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--feature-epochs", type=int, nargs="+", default=[])
    p.add_argument("--fast-dev-run", action="store_true", help="One batch per split; no checkpoint")
    return p


def run(args):
    if args.feature_epochs:
        if (args.dataset, args.arch_name, args.pretrained, args.dropout) != (
            "eurosat",
            "ResNet18",
            False,
            0.0,
        ):
            raise ValueError("Feature extraction requires scratch ResNet18 on EuroSAT, no dropout")
        if not all(1 <= epoch <= args.epochs for epoch in args.feature_epochs):
            raise ValueError("Feature epochs must fall between 1 and --epochs")
    L.seed_everything(args.seed, workers=True)
    args.lr = args.lr if args.lr is not None else (0.025 if args.dataset == "caltech101" else 0.001)
    module_class, folder = DATASETS[args.dataset]
    path = args.dataset_path or dataset_path(folder)
    dm = module_class(path, args.batch_size, args.num_workers, seed=args.seed)
    dm.setup("fit")
    rgb = args.dataset == "caltech101"
    print("Computing training-set channel statistics...", flush=True)
    stats = (fit_rgb_statistics if rgb else fit_statistics)(dm.raw_train_dataset)
    size = 112 if args.arch_name == "ViT-Tiny" else (224 if rgb else None)
    dm.train_transform = build_transforms(
        stats, train=True, image_size=size, augmentation=args.augmentation, rgb=rgb
    )
    dm.eval_transform = build_transforms(stats, train=False, image_size=size, rgb=rgb)

    network = get_network(
        args.arch_name, dm.num_channels, dm.num_classes, args.pretrained, args.dropout
    )
    model = BaseModel(args, dm, network)
    logger = CSVLogger(str(args.output_dir), name=f"{args.dataset}-{args.arch_name}")
    run_dir = Path(logger.experiment.log_dir)
    stats.to_json(run_dir / "statistics.json")
    (run_dir / "classes.json").write_text(json.dumps(dm.class_names, indent=2))
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", monitor=model.monitor, mode="max", save_top_k=1
    )
    callbacks = [checkpoint]
    if args.feature_epochs:
        callbacks.append(FeatureExtractor(run_dir / "features", tuple(args.feature_epochs)))
    else:
        callbacks.append(EarlyStopping(model.monitor, mode="max", patience=args.patience))
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm, ckpt_path=None if args.fast_dev_run else "best")[0]
    summary = {
        "checkpoint": checkpoint.best_model_path,
        "smoke_test": args.fast_dev_run,
        "metrics": results,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    print(json.dumps(run(parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()

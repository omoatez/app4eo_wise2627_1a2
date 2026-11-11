"""Inspect RGB composites or visualize saved penultimate representations."""

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import EuroSATDataset, TinyBENDataset  # noqa: E402


def plot_rgb(dataset, output: Path, count: int = 6, seed: int = 42):
    """Display B04/B03/B02 with a per-image stretch used for visualization only."""
    count = min(count, len(dataset))
    if count < 1:
        raise ValueError("At least one image is required.")
    indexes = np.random.default_rng(seed).choice(len(dataset), count, replace=False)
    columns = min(3, count)
    figure, axes = plt.subplots(
        (count + columns - 1) // columns,
        columns,
        squeeze=False,
        figsize=(5 * columns, 4 * ((count + columns - 1) // columns)),
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indexes, strict=False):
        image, label = dataset[int(index)]
        if image.dtype != np.uint16 or image.ndim != 3 or image.shape[0] != 10:
            raise ValueError("Expected raw uint16 imagery with ten CHW bands")
        rgb = image[[2, 1, 0]].astype(np.float32)
        scale = np.percentile(rgb, 99, axis=(1, 2), keepdims=True)
        rgb = np.clip(rgb / np.where(scale > 0, scale, 1), 0, 1).transpose(1, 2, 0)
        names = (
            [dataset.class_names[i] for i in np.flatnonzero(label)]
            if np.ndim(label)
            else [dataset.class_names[int(label)]]
        )
        axis.imshow(rgb)
        axis.set_title(fill(", ".join(names), width=45), fontsize=9)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_umap(source: Path, output: Path, seed: int = 42, neighbors: int = 15):
    try:
        from umap import UMAP
    except ImportError as error:
        raise SystemExit("Install analysis dependencies: pip install -e '.[analysis]'") from error
    with np.load(source, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        labels = data["labels"]
        class_names = data["class_names"]
    if features.ndim != 2 or len(features) < 4 or labels.shape != (len(features),):
        raise ValueError("UMAP requires at least four single-label feature rows")
    if not np.isfinite(features).all() or neighbors < 2:
        raise ValueError("Features must be finite and n_neighbors must be at least 2")
    embedding = UMAP(
        n_components=2,
        n_neighbors=min(neighbors, len(features) - 1),
        min_dist=0.1,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(features)
    figure, axis = plt.subplots(figsize=(9, 7))
    colors = plt.get_cmap("tab10")
    for index, name in enumerate(class_names):
        points = embedding[labels == index]
        axis.scatter(
            points[:, 0], points[:, 1], s=5, alpha=0.6, label=str(name), color=colors(index % 10)
        )
    axis.set(title=f"Training representations — {source.stem}", xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(loc="best", markerscale=3, fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    np.savez_compressed(output.with_suffix(".npz"), embedding=embedding, labels=labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rgb = commands.add_parser("rgb", help="Verify raw imagery and plot labeled RGB composites")
    rgb.add_argument("--dataset", choices=("eurosat", "tinyben"), required=True)
    rgb.add_argument("--dataset-path", type=Path, required=True)
    rgb.add_argument("--split", choices=("train", "validation", "test"), default="train")
    rgb.add_argument("--count", type=int, default=6)
    rgb.add_argument("--seed", type=int, default=42)
    rgb.add_argument("--output", type=Path, default=Path("outputs/rgb.png"))
    umap = commands.add_parser("umap", help="Plot a saved feature archive")
    umap.add_argument("features", type=Path)
    umap.add_argument("--output", type=Path, default=Path("outputs/umap.png"))
    umap.add_argument("--seed", type=int, default=42)
    umap.add_argument("--neighbors", type=int, default=15)
    args = parser.parse_args(argv)
    if args.command == "rgb":
        dataset = (
            EuroSATDataset(args.dataset_path, args.split, seed=args.seed)
            if args.dataset == "eurosat"
            else TinyBENDataset(args.dataset_path, args.split)
        )
        plot_rgb(dataset, args.output, args.count, args.seed)
    else:
        plot_umap(args.features, args.output, args.seed, args.neighbors)
    print(args.output)


if __name__ == "__main__":
    main()

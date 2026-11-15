# APP4EO — Milestone 1

General-task code with one training entry point. Start in `experiments.py`:
`main()` reads arguments → `run()` builds the data, model, and Lightning trainer.

## Setup and run

Use the existing `.venv`, or create one with Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
pytest -q
python experiments.py --dataset eurosat --arch-name ResNet-6
```

The downloaded datasets are extracted under `untracked-files/`. You can override
their location with `--dataset-path`:

```text
EuroSAT/<CLASS>/<IMAGE>.tif
tinyBEN/images/<PATCH_ID>.tif
tinyBEN/{train,val,test}.csv
caltech101/101_ObjectCategories/<CLASS>/<IMAGE>.jpg
```

Defaults match the assignment: 30 epochs, batch size 32, AdamW, OneCycleLR,
learning rate 0.001 (0.025 for Caltech), weight decay 0.01, and patience 5.
Use `--arch-name ResNet18` or `ViT-Tiny`, `--pretrained`, and `--dropout 0.5`
for other configurations. `--fast-dev-run --num-workers 0 --accelerator cpu`
checks one batch per split. Run `python experiments.py --help` for all options.

## Files

| File | Purpose |
| --- | --- |
| `experiments.py` | Assemble and run an experiment |
| `base.py` | Losses, epoch metrics, optimizer and scheduler |
| `models.py` | ResNet-6 and timm ResNet18/ViT-Tiny |
| `data/datasets.py` | Read images, map labels and create splits |
| `data/datamodules.py` | Batch data with train/evaluation transforms |
| `data/transform.py` | Training statistics, scaling and augmentation |
| `features.py` | Optional penultimate feature extraction |
| `analysis.py` | RGB previews and UMAP plots |
| `tests/` | Dataset, model and complete training checks |

Both satellite datasets use the same ten bands selected by GeoTIFF descriptions.
Scaling and normalization statistics come only from training images. EuroSAT
uses a reproducible stratified 70/10/20 split. Evaluation has no augmentation.
Training always includes flips; `--augmentation` adds one other technique.

Runs write CSV metrics, `hparams.yaml`, statistics, class names, the best checkpoint,
and test results to `outputs/<dataset>-<architecture>/version_N/`.
Normal runs test the best validation checkpoint; smoke runs use their current weights.
This foundation supports one device. `requirements-lock.txt` records the tested
macOS ARM/Python 3.11 environment; `pyproject.toml` defines portable dependencies.

## GPU Runner

Upload `dist/app4eo-milestone1.zip` and enter **`run_gpu.py`** as the entry point.
It runs EuroSAT / ResNet-6 on one GPU; edit its arguments to change the experiment.
The code uses `$DATA_DIR/untracked-files` (normally `/data/untracked-files`) on the
runner and the project-local `untracked-files/` folder locally. Datasets are already
on the server and are excluded from the ZIP, along with `.venv` and outputs.
Results go to `outputs/` in the job working directory. The portal's installed
dependencies and output-retention behavior have not been verified; dependency
requirements are in `pyproject.toml`. The allocation is limited to one hour, so
completion of a full 30-epoch run depends on the assigned GPU.

## Optional analysis

```sh
python analysis.py rgb --dataset eurosat --dataset-path untracked-files/EuroSAT
python experiments.py --dataset eurosat --arch-name ResNet18 --feature-epochs 5 10
python analysis.py umap outputs/eurosat-ResNet18/version_0/features/epoch-05.npz
```

Feature mode disables early stopping and saves float16 features without augmentation.
Real benchmark runs and the report are still pending. Topics A/B are not
implemented. The Caltech course starter was absent: its fallback uses RGB images,
excludes `BACKGROUND_Google`, and assumes a 70/10/20 split that must be reconciled
with the course starter. Check formatting with `ruff check .` and `ruff format --check .`.

to create the zip for the runner, run the following:

```sh
mkdir -p dist

rm -f dist/app4eo-milestone1.zip

zip -r dist/app4eo-milestone1.zip \
run_gpu.py \
experiments.py \
base.py \
models.py \
features.py \
analysis.py \
pyproject.toml \
README.md \
data/__init__.py \
data/datamodules.py \
data/datasets.py \
data/transform.py
```
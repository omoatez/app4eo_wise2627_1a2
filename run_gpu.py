
"""GPU Runner: EuroSAT / ResNet18 / from scratch."""

from experiments import main

if __name__ == "__main__":
    main([
        "--dataset", "eurosat",
        "--arch-name", "ResNet18",
        "--dropout", "0.5",
        "--accelerator", "gpu",
    ])
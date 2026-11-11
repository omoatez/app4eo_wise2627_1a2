"""GPU Runner ZIP entry point. Edit these arguments to choose the experiment."""

from experiments import main

if __name__ == "__main__":
    main(
        [
            "--dataset",
            "eurosat",
            "--arch-name",
            "ResNet-6",
            "--accelerator",
            "gpu",
        ]
    )

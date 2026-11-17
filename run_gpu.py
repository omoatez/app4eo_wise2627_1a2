from experiments import main

if __name__ == "__main__":
    main([
        "--dataset", "eurosat",
        "--arch-name", "ResNet18",
        "--augmentation", "brightness",
        "--accelerator", "gpu",
    ])

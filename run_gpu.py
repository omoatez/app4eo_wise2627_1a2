from experiments import main

if __name__ == "__main__":
    main([
        "--dataset", "tinyben",
        "--arch-name", "ResNet18",
        "--accelerator", "gpu",
    ])

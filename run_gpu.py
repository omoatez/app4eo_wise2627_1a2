from experiments import main

if __name__ == "__main__":
    main([
        "--dataset", "tinyben",
        "--arch-name", "ResNet-6",
        "--dropout", "0.5",
        "--accelerator", "gpu",
    ])

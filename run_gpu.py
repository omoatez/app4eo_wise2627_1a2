from experiments import main

if __name__ == "__main__":
    main([
        "--dataset", "eurosat",
        "--arch-name", "ViT-Tiny",
        "--pretrained",
        "--accelerator", "gpu",
    ])

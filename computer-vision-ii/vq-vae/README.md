# VQ-VAE

Computer Vision II, year 4 of the Mathematical Engineering and AI degree, ICAI (Fall 2025).

Vector-quantised variational autoencoder on CIFAR-10: residual encoder and decoder, a 512-entry codebook trained either with the codebook loss or with exponential moving averages.

## What it does

- **`Encoder`** and **`Decoder`**: convolutional networks with residual stacks.
- **`VectorQuantizer`**: nearest-codebook lookup with the straight-through estimator, codebook and commitment losses (van den Oord et al., 2017).
- **`VectorQuantizerEMA`** and `update_ema_weights`: the variant that updates the codebook with exponential moving averages instead of gradients.
- Reconstruction loss, and training and evaluation loops with mixed precision on GPU.

Default configuration: 512 codes of dimension 64, commitment cost 0.25, EMA decay 0.99, 15,000 updates with batch size 256.

## Run

```bash
uv sync
uv run python -m src.train    # downloads CIFAR-10 on first run
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.

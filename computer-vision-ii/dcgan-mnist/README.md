# DCGAN on MNIST

Computer Vision II, year 4 of the Mathematical Engineering and AI degree, ICAI (Fall 2025).

Deep convolutional GAN written from scratch: convolutional discriminator, transposed-convolution generator, their adversarial losses and the training loop on MNIST.

## What it does

- **`Discriminator`**: strided convolutions with batch normalisation and LeakyReLU, and a linear output.
- **`Generator`**: linear projection of the noise vector to a 7x7 feature map and transposed convolutions up to 28x28.
- Discriminator and generator losses with `BCEWithLogitsLoss`, and the alternating training loop, which saves samples and loss curves.

## Run

```bash
uv sync
uv run python -m src.train    # downloads MNIST on first run
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.

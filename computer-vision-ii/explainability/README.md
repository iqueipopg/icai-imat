# Explainability for image classifiers

Computer Vision II, year 4 of the Mathematical Engineering and AI degree, ICAI (Fall 2025).

Saliency maps, occlusion, guided backpropagation and integrated gradients on a ResNet-50, plus a pixel-removal test of how faithful each explanation is.

## What it does

- **Saliency maps**: gradient of the class score with respect to the input pixels.
- **Occlusion** (Zeiler and Fergus, 2014): drop in the class score when a sliding patch is masked.
- **Guided backpropagation** (Springenberg et al., 2014): backward hooks on the ReLUs that only let positive gradients through.
- **Integrated gradients** (Sundararajan et al., 2017): gradients averaged along the path from a baseline to the image.
- **`Visualizer`**: plots each explanation and removes the most relevant pixels to check how much the prediction depends on them.

## Run

```bash
uv sync
uv run python -m src.example    # explains a ResNet-50 prediction on a sample image
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.

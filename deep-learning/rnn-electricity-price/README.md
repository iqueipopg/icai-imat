# RNN with a hand-written backward pass for day-ahead electricity prices

Deep Learning, year 3 of the Mathematical Engineering and AI degree, ICAI (Spring 2025).

Forward and backward pass of an RNN written as a custom `torch.autograd.Function`, and an LSTM that forecasts the 24 hourly day-ahead prices of the Nord Pool market.

## What it does

- **`RNNFunction`**: a `torch.autograd.Function` that computes the forward pass of a ReLU RNN and its gradients by backpropagation through time, written by hand instead of relying on autograd.
- **`ElectricDataset`**: turns the hourly price series into samples of `past_days x 24` hours and the 24 prices of the next day as the target.
- **`MyModel`**: a five-layer LSTM with dropout and a linear head that outputs the 24 prices of the next day.
- Training and evaluation loops on the standardised series; the evaluation metric is the test MAE.

The data is the Nord Pool (NP) day-ahead price dataset from the open-access benchmark of Lago et al. (2021), downloaded from [Zenodo](https://zenodo.org/records/4624805) on first run.

## Run

```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.

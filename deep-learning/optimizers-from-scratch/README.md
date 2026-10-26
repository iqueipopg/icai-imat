# Optimizers from scratch

Deep Learning, year 3 of the Mathematical Engineering and AI degree, ICAI (Spring 2025).

SGD, momentum, Nesterov and Adam written as `torch.optim.Optimizer` subclasses, plus a step learning-rate scheduler and dropout, used to fine-tune a ResNet-18 on Imagenette.

## What it does

The update rules of four optimizers are written by hand against the `torch.optim.Optimizer` interface, without calling PyTorch's implementations:

- **SGD** with weight decay.
- **SGD with momentum** and **Nesterov momentum**.
- **Adam**, with bias-corrected first and second moments.
- A **StepLR** scheduler (`torch.optim.lr_scheduler.LRScheduler` subclass) and a **Dropout** module.

The model is a ResNet-18 pretrained on ImageNet, with its classification head replaced for the 10 Imagenette classes and an option to unfreeze its last residual stages. The training loop logs loss and accuracy to TensorBoard.

## Run

```bash
pip install -r requirements.txt
python -m src.train      # downloads Imagenette (fast.ai) into data/ on first run
python -m src.evaluate
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.

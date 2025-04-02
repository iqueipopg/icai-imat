# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO
    model.to(device)
    model.train()
    loss_run: float = 0.0

    for idx, (inputs, targets) in enumerate(tqdm(train_data, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)

        # Normalize targets
        targets = ((targets - mean) / std).float()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        batch_loss = loss(outputs, targets)
        batch_loss.backward()
        optimizer.step()

        loss_run += batch_loss.item()

        if idx % 10 == 0:  # Log every 10 batches
            writer.add_scalar(
                "Training Loss",
                batch_loss.item(),
                epoch * len(train_data) + idx,
            )

    avg_loss: float = loss_run / len(train_data)
    writer.add_scalar("Epoch Training Loss", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO
    model.to(device)
    model.eval()
    loss_run: float = 0.0

    for idx, (inputs, targets) in enumerate(tqdm(val_data, desc="Validation")):
        inputs, targets = inputs.to(device), targets.to(device)

        # Normalize targets
        targets = ((targets - mean) / std).float()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        batch_loss = loss(outputs, targets)
        loss_run += batch_loss.item()

    avg_loss: float = loss_run / len(val_data)
    writer.add_scalar("Validation Loss", avg_loss, epoch)

    if scheduler is not None:
        scheduler.step(avg_loss)

    return avg_loss


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    # TODO
    model.to(device)
    model.eval()
    mae: float = 0.0

    for inputs, targets in tqdm(test_data, desc="Testing"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Normalize targets
        targets = ((targets - mean) / std).float()

        # Forward pass
        outputs = model(inputs)
        outputs = outputs * std + mean
        targets = targets * std + mean

        # Calculate mae
        batch_mae = torch.mean(torch.abs(outputs - targets))
        mae += batch_mae.item()

    avg_mae: float = mae / len(test_data)
    return avg_mae

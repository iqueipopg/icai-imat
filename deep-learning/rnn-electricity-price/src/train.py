# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step, t_step
from src.utils import set_seed, save_model

# static variables
DATA_PATH: Final[str] = "data"
PAST_DAYS: Final[int] = 7
BATCH_SIZE: Final[int] = 64
EPOCHS: Final[int] = 10
LEARNING_RATE: Final[float] = 0.001
MODEL_NAME: Final[str] = "best_model"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO
    train_loader, val_loader, test_loader, mean, std = load_data(
        save_path=DATA_PATH,
        past_days=PAST_DAYS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # Initialize model
    model = MyModel(hidden_size=128).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Training step
        train_loss = train_step(
            model=model,
            train_data=train_loader,
            mean=mean,
            std=std,
            loss=criterion,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            device=device,
        )

        # Validation step
        val_loss = val_step(
            model=model,
            val_data=val_loader,
            mean=mean,
            std=std,
            loss=criterion,
            scheduler=None,
            writer=writer,
            epoch=epoch,
            device=device,
        )

        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

    # Save model
    save_model(model, MODEL_NAME)
    print("Model saved to models/best_model.pt")

    test_mae = t_step(
        model=model,
        test_data=test_loader,
        mean=mean,
        std=std,
        device=device,
    )
    print(f"Test MAE: {test_mae:.4f}")

    writer.close()


if __name__ == "__main__":
    main()

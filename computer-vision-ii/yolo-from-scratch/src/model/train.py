"""
Script to train the model.
"""

import os
from typing import cast

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import BATCH_SIZE, EPOCHS, LR, PATH_MODEL
from src.data.dataset import RaccoonDataset
from src.model.loss import YOLOLoss
from src.model.model import YOLO
from src.utils import mean_average_precision


class Trainer:
    """
    Class of the trainer of the model.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        self.train_loader, self.test_loader = self._get_loaders()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = YOLOLoss()
        self.losses: dict[str, list[float | tuple[int, float]]] = {
            "coord": [],
            "obj": [],
            "noobj": [],
            "class": [],
            "loss": [],
            "map_train": [],
            "map_test": [],
        }

    def _get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Obtains the train and test loaders.

        Returns:
            Train and test loader.
        """
        # TODO
        from torch.utils.data import Subset

        train_dataset = RaccoonDataset(
            img_dir="data/images_resized/train", label_dir="data/labels_resized/train"
        )

        test_dataset_full = RaccoonDataset(
            img_dir="data/images_resized/test", label_dir="data/labels_resized/test"
        )

        # Create a subset of test dataset with exactly 32 samples
        test_dataset = Subset(test_dataset_full, range(32))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, test_loader

    def _append_losses(
        self,
        coord_loss: torch.Tensor,
        obj_loss: torch.Tensor,
        noobj_loss: torch.Tensor,
        class_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Appends the losses of the batch to the dictionary of the losses.

        Args:
            coord_loss: Loss for the predicted coordinates.
            obj_loss: Loss for the prediction of if an object exists in the cell.
            noobj_loss: Loss for the prediction of if an object does not exist in the
                cell.
            class_loss: Loss for the prediction of the class of the object.
            loss: Total loss.
        """

        self.losses["coord"].append(float(coord_loss))
        self.losses["obj"].append(float(obj_loss))
        self.losses["noobj"].append(float(noobj_loss))
        self.losses["class"].append(float(class_loss))
        self.losses["loss"].append(float(loss))

    def _make_epoch_train(self) -> None:
        """
        Performs one epoch of the training.
        """
        # TODO
        self.model.train()

        for batch_idx, (images, targets) in enumerate(
            tqdm(self.train_loader, desc="Training")
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(images)

            coord_loss, obj_loss, noobj_loss, class_loss, total_loss = self.criterion(
                predictions, targets
            )
            torch.autograd.set_detect_anomaly(True)

            total_loss.backward()

            self.optimizer.step()

            self._append_losses(
                coord_loss, obj_loss, noobj_loss, class_loss, total_loss
            )

    def fit(
        self,
        n_epochs: int = EPOCHS,
        start_epochs: list[int] | None = None,
        save_every: int = 10,
        path_weights: str = PATH_MODEL,
        path_images: str | None = "images/training",
    ) -> None:
        """
        Training of the network. It saves the parameters of the trained model.

        Args:
            n_epochs: Number of epochs to train.
            start_epochs: List with the starting epoch for each graphic saved.
            save_every: To save the model and calculate mAP every 'save_every' epochs.
            path_weights: Path where the weights are saved.
            path_images: Path where we save the images of the evolution of the training.
        """
        # TODO
        print(f"Starting training for {n_epochs} epochs.")

        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch} of {n_epochs}.")
            self._make_epoch_train()

            if epoch % save_every == 0:
                print(f"Saving weights and evaluating metrics at epoch {epoch}.")

                os.makedirs(os.path.dirname(path_weights), exist_ok=True)
                torch.save(self.model.state_dict(), path_weights)

                train_map = mean_average_precision("train", self.model)
                test_map = mean_average_precision("test", self.model)

                self.losses["map_train"].append((epoch, train_map))
                self.losses["map_test"].append((epoch, test_map))

                print(f"Metrics -> Train: {train_map:.4f}, Test: {test_map:.4f}")

                if path_images:
                    self._plot_training_evolution(start_epochs, path_images)

            if self.losses["loss"]:
                print(f"Current loss: {self.losses['loss'][-1]:.4f}")

        # Save final weights
        os.makedirs(os.path.dirname(path_weights), exist_ok=True)
        torch.save(self.model.state_dict(), path_weights)
        print(f"Training completed, model saved to {path_weights}.")

        if path_images:
            self._plot_training_evolution(start_epochs, path_images)

    def test(self) -> float:
        """
        Test of the trained network.

        Returns:
            Loss in the test set.
        """
        # TODO
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images)

                coord_loss, obj_loss, noobj_loss, class_loss, loss = self.criterion(
                    predictions, targets
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def _save_graphic_loss(self, start_epoch: int, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            start_epoch: First epoch of the graphic.
            path_images: Path where the images are saved.
        """

        x = range(start_epoch + 1, len(self.losses["loss"]) + 1)

        fig, axs = plt.subplots()
        axs.plot(x, self.losses["coord"][start_epoch:], label="Coordinates")
        axs.plot(x, self.losses["obj"][start_epoch:], label="Objects")
        axs.plot(x, self.losses["noobj"][start_epoch:], label="No Objects")
        axs.plot(x, self.losses["class"][start_epoch:], label="Classification")
        axs.plot(x, self.losses["loss"][start_epoch:], label="Total")

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Loss")
        axs.set_title("Evolution of the Loss During the Training")

        # Create directory if it doesn't exist
        save_path = f"{path_images}_loss_{start_epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    def _save_graphic_map(self, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            path_images: Path where the images are saved.
        """

        map_train = cast(list[tuple[int, float]], self.losses["map_train"])
        map_test = cast(list[tuple[int, float]], self.losses["map_test"])

        x = [epoch for epoch, _ in map_train]

        fig, axs = plt.subplots()
        axs.plot(x, [value for _, value in map_train], label="Train")
        axs.plot(x, [value for _, value in map_test], label="Test")

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("mAP")
        axs.set_title("Evolution of the mAP During the Training")

        # Create directory if it doesn't exist
        save_path = f"{path_images}_map.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    def _plot_training_evolution(
        self, start_epochs: list[int] | None = None, path_images: str | None = None
    ) -> None:
        """
        Plots the evolution of the training loss.

        Args:
            start_epochs: List with the starting epoch for each graphic saved.
            path_images: Path where the images are saved.
        """

        if start_epochs is None:
            total_epochs = len(self.losses["loss"])
            if total_epochs < 50:
                start_epochs = [0]
            elif total_epochs < 100:
                start_epochs = [0, 50]
            else:
                start_epochs = [0, 50, 100]
        if path_images is None:
            path_images = "images/training/evolution"

        for start_epoch in start_epochs:
            self._save_graphic_loss(start_epoch, path_images=path_images)

        self._save_graphic_map(path_images)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fit()

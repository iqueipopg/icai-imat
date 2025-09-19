"""
Script to train the model.
"""

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

    def fit(
        self,
        n_epochs: int = EPOCHS,
        start_epochs: list[int] | None = None,
        save_every: int = 5,
        path_weights: str = PATH_MODEL,
        path_images: str | None = None,
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

    def test(self) -> float:
        """
        Test of the trained network.

        Returns:
            Loss in the test set.
        """

        # TODO

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

        fig.savefig(f"{path_images}_loss_{start_epoch}.png")
        plt.close(fig)

    def _save_graphic_map(self, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            path_images: Path where the images are saved.
        """

        x = [xx[0] for xx in self.losses["map_train"]]  # type: ignore

        fig, axs = plt.subplots()
        axs.plot(
            x, [xx[1] for xx in self.losses["map_train"]], label="Train"  # type: ignore
        )
        axs.plot(
            x, [xx[1] for xx in self.losses["map_test"]], label="Test"  # type: ignore
        )

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("mAP")
        axs.set_title("Evolution of the mAP During the Training")

        fig.savefig(f"{path_images}_map.png")
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

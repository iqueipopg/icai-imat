# deep learning libraries
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm

# own modules
from src.models import CNNModel
from src.data import load_data
from src.utils import (
    Accuracy,
    save_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"

NUMBER_OF_CLASSES: int = 10

accuracy_calculator = Accuracy()


import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    Ejecuta una iteración de entrenamiento.

    Args:
        model: Modelo de red neuronal.
        train_data: DataLoader con los datos de entrenamiento.
        loss: Función de pérdida.
        optimizer: Optimizador utilizado.
        writer: Escritor de logs de TensorBoard.
        epoch: Número de la época actual.
        device: Dispositivo de cómputo (CPU/GPU).
    """

    # Listas para almacenar métricas
    loss_values: list[float] = []
    accuracy_values: list[float] = []

    model.train()  # Activar modo de entrenamiento

    for batch_inputs, batch_targets in tqdm(train_data):
        # Enviar los datos al dispositivo adecuado
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # Reiniciar los gradientes del optimizador
        optimizer.zero_grad()

        # Propagación hacia adelante
        batch_outputs = model(batch_inputs)

        # Cálculo de la pérdida
        batch_loss = loss(batch_outputs, batch_targets)

        # Retropropagación y actualización de pesos
        batch_loss.backward()
        optimizer.step()

        # Cálculo de precisión
        accuracy_calculator.reset()
        accuracy_calculator.update(logits=batch_outputs, labels=batch_targets)
        batch_accuracy = accuracy_calculator.compute()

        # Guardar las métricas de la iteración
        loss_values.append(batch_loss.item())
        accuracy_values.append(batch_accuracy)

    # Registrar métricas en TensorBoard
    writer.add_scalar("train/loss", np.mean(loss_values), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracy_values), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    Ejecuta una iteración de validación.

    Args:
        model: Modelo de red neuronal.
        val_data: DataLoader con los datos de validación.
        loss: Función de pérdida.
        writer: Escritor de logs de TensorBoard.
        epoch: Número de la época actual.
        device: Dispositivo de cómputo (CPU/GPU).
    """

    # Listas para métricas
    loss_values: list[float] = []
    accuracy_values: list[float] = []

    model.eval()  # Activar modo de evaluación

    # Deshabilitar gradientes para validación
    with torch.no_grad():
        for batch_inputs, batch_targets in tqdm(val_data):
            # Enviar los datos al dispositivo adecuado
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(
                device
            )

            # Propagación hacia adelante
            batch_outputs = model(batch_inputs)

            # Cálculo de la pérdida
            batch_loss = loss(batch_outputs, batch_targets)

            # Cálculo de precisión
            accuracy_calculator.reset()
            accuracy_calculator.update(logits=batch_outputs, labels=batch_targets)
            batch_accuracy = accuracy_calculator.compute()

            # Guardar métricas de la iteración
            loss_values.append(batch_loss.item())
            accuracy_values.append(batch_accuracy)

    # Registrar métricas en TensorBoard
    writer.add_scalar("val/loss", np.mean(loss_values), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracy_values), epoch)


def main() -> None:
    """
    Función principal que ejecuta el proceso de entrenamiento.
    """

    # Definición de hiperparámetros
    hidden_sizes = (64, 128, 256)
    batch_size = 64
    lr = 1e-3
    epochs = 1
    dropout_prob = 0.35
    dataloader_workers = 8
    unfreezed_layers = 2

    # Limpiar logs anteriores
    open("nohup.out", "w").close()

    # Cargar datos de entrenamiento y validación
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(
        DATA_PATH, batch_size=batch_size, num_workers=dataloader_workers
    )

    # Crear nombre de experimento y escritor de logs
    model_name: str = (
        f"model_lr_{lr}_pretrained_{unfreezed_layers}_{batch_size}_{epochs}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{model_name}")

    # Definir modelo y enviarlo al dispositivo adecuado
    model: torch.nn.Module = CNNModel(
        num_classes=NUMBER_OF_CLASSES,
        freeze_base=True,
        unfreeze_layers=unfreezed_layers,
    ).to(device)

    # Definir función de pérdida y optimizador
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Bucle de entrenamiento
    for epoch in range(epochs):
        print(f"Época {epoch+1}:")

        # Ejecutar paso de entrenamiento
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # Ejecutar paso de validación
        val_step(model, val_data, loss, writer, epoch, device)

    # Guardar el modelo entrenado
    save_model(model, model_name)

    return None


if __name__ == "__main__":
    main()

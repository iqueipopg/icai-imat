import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader
import numpy as np
from src.data import load_data
from src.utils import Accuracy, set_seed

# Set device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set all seeds and number of threads
set_seed(42)
torch.set_num_threads(8)

# Static variable for data path
DATA_PATH: str = "data"

# Accuracy calculator instance
accuracy_calculator = Accuracy()


def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluates the model performance on the test set.

    Args:
        model: PyTorch model to evaluate.
        test_data: DataLoader for the test dataset.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        The average accuracy over the test dataset.
    """
    accuracies: list[float] = []

    model.eval()  # Set the model to evaluation mode

    # No gradient computation needed during evaluation
    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Update accuracy
            accuracy_calculator.reset()
            accuracy_calculator.update(logits=outputs, labels=targets)
            batch_accuracy = accuracy_calculator.compute()

            # Append batch accuracy
            accuracies.append(batch_accuracy)

    # Calculate and return average accuracy
    avg_accuracy = torch.mean(torch.tensor(accuracies)).item()
    return avg_accuracy


def main(name: str) -> float:
    """
    Main function to run the model evaluation.

    Args:
        name: The name of the saved model to evaluate.

    Returns:
        Accuracy of the evaluated model.
    """
    # Load test data
    _, _, test_data = load_data(DATA_PATH, batch_size=32, num_workers=8)

    print(f"Loading model {name}...")
    try:
        model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)
    except FileNotFoundError:
        print(f"Model {name} not found.")
        return 0.0  # Return an appropriate error value or 0.0 accuracy

    # Evaluate model
    accuracy: float = t_step(model, test_data, device)
    return accuracy


if __name__ == "__main__":
    print("Evaluating model...")
    model_name = "best_model"
    accuracy = main(model_name)
    print(f"Accuracy of {model_name}: {accuracy:.4f}")

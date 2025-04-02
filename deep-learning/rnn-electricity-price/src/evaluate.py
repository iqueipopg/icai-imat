import torch
from src.data import load_data
from src.utils import set_seed, load_model
from src.train_functions import t_step

# Set device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# static variables
DATA_PATH: str = "data"
PAST_DAYS: int = 7
BATCH_SIZE: int = 64
MODEL_NAME: str = "best_model"

# Set all seeds and number of threads
set_seed(42)
torch.set_num_threads(8)


def main(name: str) -> float:
    """
    Main function to run the model evaluation.

    Args:
        name: The name of the saved model to evaluate.

    Returns:
        Accuracy of the evaluated model.
    """
    # Load test data
    _, _, test_data, mean, std = load_data(
        save_path=DATA_PATH,
        past_days=PAST_DAYS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    print(f"Loading model {name}...")
    # Load the model
    model = load_model(MODEL_NAME).to(device)
    model.eval()

    test_mae = t_step(model, test_data, mean, std, device)
    return test_mae


if __name__ == "__main__":
    print("Evaluating model...")
    model_name = "best_model"
    mae = main(MODEL_NAME)
    print(f"Test Mae of {MODEL_NAME}: {mae:.4f}")

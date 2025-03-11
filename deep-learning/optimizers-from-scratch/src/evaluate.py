# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# own modules
from src.data import load_data
from src.utils import (
    Accuracy,
    load_model,
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


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")

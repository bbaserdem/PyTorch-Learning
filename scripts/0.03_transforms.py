import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

DATA_DIR = "data/00_mnist"
MODEL_PATH = f"{DATA_DIR}/00_mnist.pth"

ds = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
    ),
)

# ToTensor makes a PIL image (or numpy ndarray) into a FloatTensor and clamps to [0, 1]
# Lambda applies lambda to the output, this one encodes index to one-hot vectior.

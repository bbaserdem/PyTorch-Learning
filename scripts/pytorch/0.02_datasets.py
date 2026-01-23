import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import ToTensor

# torch.utils.data.Dataset is a class for storing samples and labels
# torch.utils.data.DataLoader wraps an iterable around a Dataset so it's easier to query

DATA_DIR = "data/00_mnist"
MODEL_PATH = f"{DATA_DIR}/00_mnist.pth"

training_data = datasets.FashionMNIST(
    root=DATA_DIR,  # Directory of dataset
    train=True,  # Whether this is the training set or not
    download=True,  # Download if local copy not found
    transform=ToTensor(),  # Specifies a transform function
)
test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=ToTensor(),
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # Get a random sample id
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # Querying by id returns a tuple of input and label tensors
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Custom dataset implementation (we are using FashionMNIST as example)


# Inherit Dataset class
class CustomImageDataset(Dataset):
    # Load some variables to class fields on initialization
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Implement a len method, number of samples in dataset
    def __len__(self):
        return len(self.img_labels)

    # Return a sample given idx
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# This class fetches samples one by one, we want to wrap it with DataLoader for iterable
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through batches
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Sample shape: {train_features[0].size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

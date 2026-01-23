import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# The neural network subclasses nn.Module, and initializes layers in __init__
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Function to turn grayscale [1, h, w] image into a vector
        # nn.Flatten() retains the minibatch dimension,
        # but reduces other dimensions to one index
        self.flatten = nn.Flatten()
        # nn.Linear is a basic linear transform, matrix multip and biases
        # nn.ReLU is the basic activation function
        # nn.Sequential is ordered container of modules
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Load the network to the device
model = NeuralNetwork().to(device)
print(model)

# A forward pass consists of the following
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Output: {logits}")
print(f"Softmax: {pred_probab}")
print(f"Predicted class: {y_pred}")

# We can probe the model structure
# Each layer with trainable weights get a weight and bias term
# Accessible through linear_relu_stack.<n>.{bias,weight}
for name, param in model.named_parameters():
    print(f"Layer: {name}\nSize: {param.size()}\nValues: {param[:2]}\n")

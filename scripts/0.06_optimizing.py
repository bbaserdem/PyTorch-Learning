import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DATA_DIR = "data/00_mnist"

# Hyperparameters are set outside the neural network
learning_rate = 1e-3
batch_size = 64
epochs = 10
hidden_layer_size = 512

training_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# OK the reason why we don't softmax is becaue nn.CrossEntropyLoss() does it automatically
# Also the reason why the labels don't need transforming
loss_fn = nn.CrossEntropyLoss()

# We will use SGD to do weight optimization (i.e. learning)
# This links this optimizer to the weights inside the model instance
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Inside the training loop;
# - Calling optimizer.zero_grad() to flush the gradients, since we do SGD, not GD
# - Backprop using loss.backwards()
# - Then optimizer.step() will adjust parameters with the accumulated grads
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Setting model to training mode so that batch norm and dropout are applied
    # Not necessary here though
    model.train()

    # run through each batch, dataloader gives the tuple of input and label
    for batch, (X, y) in enumerate(dataloader):
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Reporting
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set to just eval mode, turn off batch norm and dropout
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # We want to turn off grad computation for forward pass
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n- Accuracy: {(100 * correct):>0.1f}%\n- Avg loss: {test_loss:>8f}\n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

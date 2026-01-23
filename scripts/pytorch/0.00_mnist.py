import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DATA_DIR = "data/00_mnist"
MODEL_PATH = f"{DATA_DIR}/00_mnist.pth"

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# Define a model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # This is the architecture of our model
        # Two hidden layers with size 512
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # Flatten the input image
        x = self.flatten(x)
        # Run the flattened vector through network inference
        logits = self.linear_relu_stack(x)
        return logits


# Initialize an instance of our model
model = NeuralNetwork().to(device)
print(model)

# Define a loss function, don't know why we don't have the softmax function though
# Loss function and leaning rule is abstracted away from the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Define one epoch of training over the full dataset, in batches (SGD)
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Tell the model that it will train
    # probably makes it so that the forward pass retains activations for grad calc
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Send out input tensor and labels to the device
        X, y = X.to(device), y.to(device)

        # Compute the prediction error, runs the forward function then computes loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop; automatically defined from the forward prop I assume
        loss.backward()
        # Learn once using gradients
        optimizer.step()
        # What is happening here? Probably cleans gradients but why do this manually?
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Define a testing run
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Prime the model just for eval, so probably don't remember activations
    model.eval()

    # Evaluate in batches, probably to not overload the VRAM
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Forward pass
            pred = model(X)
            # Calculate loss
            test_loss += loss_fn(pred, y).item()
            # Calculate prediction accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Do training now
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Save our model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved PyTorch Model State to {MODEL_PATH}")

# To load a model, we need to first recreate the structure, then load the weights on it
model_load = NeuralNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
print(f"Loaded PyTorch Model from {MODEL_PATH}")

# Make predictions now
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Run prediction using the model
model_load.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        # Initialize using the inherited module
        super(Net, self).__init__()
        # Define kernels; the first two numbers are input/output features
        # Input layer features are going to be 1 for grayscale and 3 for rgb
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define the feed forward layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # How did we get 16 and 5*5 ?
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # 1: Convolution 1: 1 input image channel, 6 output channels
        # 5x5 square convolution, and use ReLU
        # Takes is (N, 1, 32, 32) and returns (N, 6, 28, 28)
        c1 = F.relu(self.conv1(input))
        # 2: Subsampling layer on 2x2 grid
        # (N, 6, 32, 32) -> (N, 6, 14, 14)
        s2 = F.max_pool2d(c1, (2, 2))
        # 3: Convolution 2, 16 output channels, 5x5 square convolution
        # (N, 6, 14, 14) -> (N, 16, 10, 10)
        c3 = F.relu(self.conv2(s2))
        # 4: Max pool with 2x2 again
        # (N, 16, 10, 10) -> (N, 16, 5, 5)
        s4 = F.max_pool2d(c3, (2, 2))
        # Flatten the spatial here for feed forward layers
        # (N, 16, 5, 5) -> (N, 16 * 5 * 5 = 400)
        s4 = torch.flatten(s4, 1)
        # FF layers
        # 5: FF
        # (N, 400) -> (N, 120)
        f5 = F.relu(self.fc1(s4))
        # 6: FF
        # (N, 120) -> (N, 84)
        f6 = F.relu(self.fc2(f5))
        # 7: FF
        # (N, 84) -> (N, 10)
        f7 = F.relu(self.fc3(f6))
        # Return the output
        return f7


net = Net()
print(f"Network architecture:\n{net}")
for i, par in enumerate(list(net.parameters())):
    print(f"Network parameter {i} size: {par.size()}")

# Try running the network on a random 32x32 input
input = torch.rand(1, 1, 32, 32)
out = net(input)
print(f"Network output on random input: {out}")

# Propagate random error signal through the network
net.zero_grad()
print(f"Backpropagate random error signal: {out.backward(torch.randn(1, 10))}")
# Why am I getting None here?

# Try again
output = net(input)  # Eval the network again
target = torch.randn(10)  # Create dummy target
target = target.view(1, -1)  # Reshape to output
criterion = nn.MSELoss()  # Establish mean square error
loss = criterion(output, target)
print(f"Loss: {loss}")

# Backprop
# Clear gradients
net.zero_grad()
print(f"Conv1 bias gradient before backprop:\n{net.conv1.bias.grad}\n")
loss.backward()
print(f"Conv1 bias gradient after  backprop:\n{net.conv1.bias.grad}\n")

# Update rule, simplest is SGD
learning_rate = 1e-2
# SGD
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
# But these are already implemented in torch.optim
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# One training loop would be
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

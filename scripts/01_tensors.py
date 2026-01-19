import numpy as np
import torch

# Normal arrays can be cast into torch tensors
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Numpy arrays can be cast into torch tensors too
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Initialization methods
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# Can generate using the shape tuple
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Can put tensor on accelerator
tensor = torch.rand(3, 4)
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Can splice just like numpy
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(f"Modded second column: {tensor[:, 1]}")

# Can concatanate using torch functions
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)

# Elementwise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z2)
print(z3)

# Can reduce a scalar to a python numerical value using item()
agg = tensor.sum()
agg_item = agg.item()
print(agg, type(agg))
print(agg_item, type(agg_item))

# In place operators modify the full array, they are prefixed with underscore
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Numpy bridge: if on CPU, numpy arrays and tenors share memory (work both ways)
t = torch.ones(5)
n = t.numpy()
print(f"t: {t}")
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
nn = np.ones(5)
tt = torch.from_numpy(nn)
print(f"tt: {tt}")
print(f"nn: {nn}")
np.add(nn, 1, out=nn)
print(f"tt: {tt}")
print(f"nn: {nn}")

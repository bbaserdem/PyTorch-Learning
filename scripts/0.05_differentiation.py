import torch

# Backprop requires knowledge of the gradient
# PyTorch has built-in differentiation engine torch.autograd
# It automatically calculates the gradient from any computational graph
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3)
w.requires_grad_(True)
b = torch.randn(3)
b.requires_grad_(True)
z = torch.matmul(x, w) + b
# We define the most basic linear transform and the cross entropy loss
# loss = CE(y, w*x + b)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Function applied to tensors to construct a computational grahp is an object of class Function
# I assume all basic math functions * + @ etc. have been defined with the backward pass
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# The loss has to be propagated, then can be retrieved through .grad fields
# Only leaf nodes with requires_grad = True will have .grad available
# backward() calculates gradients only once per graph.
# if multiple recalc are needed, need to call backward() with retain_graph = True
loss.backward()
print(w.grad)
print(b.grad)

# For just evals, we can turn off gradient computation for performance in a torch.no_grad() block
z = x @ w + b
print(f"z requires grad?: {z.requires_grad}")

with torch.no_grad():
    z = x @ w + b
print(f"z requires grad?: {z.requires_grad}")

# Or with detach
z = x @ w + b
z_det = z.detach()
print(z_det.requires_grad)
print(f"z requires grad?: {z.requires_grad}")

# Behind the scenes, autograd keeps record of exeucuted operations in a DAG of Function objects
# Tracing the graph from roots to leaves, gradients can be calculated easily
# DAG's are DYNAMIC; each .backward() call autograd starts recording a new graph

# Sometimes, output is an arbitrary tensor (cough RNNs?)
# For a vector function, may need the jacobian
inp = torch.eye(4, 5)
inp.requires_grad_(True)
out = (inp + 1).pow(2).t()
print(f"\nInput:\n{inp}")
print(f"\nOutput:\n{out}")

# I'm getting a "grad can be implicitly created only for scalar outputs" error
# So this block doesn't work
# Tracks with backward() being eqv to backward(torch.tensor(1.0))
#
# out.backward()
# print(f"\nGradient:\n{inp.grad}")
# inp.grad.zero_()

# PyTorch allows to calculate Jacobian product instead
# v.T @ J for a given input vector v
# This is done by calling .backward with an argument v
# Must be the size of the original tensor
# If output is not a scalar, the backward pass needs to be called with a tensor
# of the same size as output
vec = torch.ones_like(out)
print(f"\nJacobian product v:\n{vec}")
out.backward(vec, retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(vec, retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(vec, retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

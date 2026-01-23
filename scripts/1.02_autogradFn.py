import math

import torch

# Want to define a new function, that can backpropagate
# We will approximate y = six(x) with a legendre polynomial;
# y = a + b * P_3( c + d * x )


class LegendrePolynomial_3(torch.autograd.Function):
    """
    The implementation of custom functions is done by subclassing
    torch.autograd.Function, and implementing the forward and backward passes
    They should operate on the tensors
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, a tensor containing the input is received.
        A tensor containing the output should be returned
        ctx: context object that can be u used to stash info for backward pass
             this caching happens with ctx.save_for_backward method
        Other objects can be stored directly as attributes to ctx object
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input**3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, Tensor containing gradient of the loss wrt output
        is received; grad_output. We need to compute the gradient of the loss
        w.r.t. the input.
        We will need the original values of the input, hence we need to access ctx
        """
        (input,) = ctx.saved_tensors
        return grad_output * 0.5 * (5 * 3 * input**2 - 3)


dtype = torch.float
device = torch.device("cpu")

# Create the input and output tensors
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random tensors for weights; we need 4 here.
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Functions are applied using Function.apply
    P3 = LegendrePolynomial_3.apply

    # Forward pass
    y_pred = a + b * P3(c + d * x)

    # Calculate loss, a scalar tensor
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(f"Current loss: {loss.item()}")

    # Use autograd to propagate the error
    loss.backward()

    # Update weights
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        # Zero the grads
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"Result: y = {a.item()} + {b.item()} * P_3({c.item()} + {d.item()} x)")

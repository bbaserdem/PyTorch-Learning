import math
import random

import torch


class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate five parameters
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        A weird function, where the number 4 or 5 is randomly chosen.
        e parameter is reused to compute the contribution of the orders.

        Each forward pass stores the dynamic computational graph.
        So normal control-flow operators can be used

        Same parameter can be reused multiple times safely.
        """
        y = self.a + self.b * x + self.c * x**2 + self.d * x**3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x**exp
        return y

    def string(self):
        """
        Can do custom methods like the rest of python
        """
        return f"y = {self.a.item():.3g} + {self.b.item():.3g} x + {self.c.item():.3g} x^2 + {self.d.item():.3g} x^3 + {self.e.item():.3g} x^4 ? x^5"


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = DynamicNet()

criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.8)
for t in range(30000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(f"Iteration {t}\nLoss: {loss.item()}\nFunction: {model.string()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"\nResult: {model.string()}")

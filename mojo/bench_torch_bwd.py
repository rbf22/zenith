import time

import torch

from zenith.mamba3 import ssd

B, T, H, P, N, L = 4, 512, 20, 64, 128, 64
dev = "mps"


def inputs():
    x = torch.randn(B, T, H, P, device=dev, requires_grad=True)
    xp = torch.randn(B, T, H, P, device=dev, requires_grad=True)
    q, k, kp = (torch.randn(B, T, H, N, device=dev, requires_grad=True) for _ in range(3))
    adt = (-0.05 * torch.rand(B, T, H, device=dev)).requires_grad_()
    gamma = (0.05 * torch.rand(B, T, H, device=dev)).requires_grad_()
    beta = (0.05 * torch.rand(B, T, H, device=dev)).requires_grad_()
    return x, q, k, kp, xp, adt, gamma, beta


args = inputs()


def step():
    y = ssd(*args, L)
    y.sum().backward()


for _ in range(3):
    step()
torch.mps.synchronize()
iters = 10
t0 = time.time()
for _ in range(iters):
    step()
torch.mps.synchronize()
print(f"torch MPS ssd fwd+bwd ms/pair: {(time.time() - t0) / iters * 1000:.2f}")

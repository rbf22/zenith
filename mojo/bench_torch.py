import time

import torch

from zenith.mamba3 import ssd

B, T, H, P, N, L = 4, 512, 20, 64, 128, 64
dev = "mps"
x = torch.randn(B, T, H, P, device=dev)
xp = torch.randn(B, T, H, P, device=dev)
q, k, kp = (torch.randn(B, T, H, N, device=dev) for _ in range(3))
adt = -0.05 * torch.rand(B, T, H, device=dev)
gamma = 0.05 * torch.rand(B, T, H, device=dev)
beta = 0.05 * torch.rand(B, T, H, device=dev)

for _ in range(3):
    ssd(x, q, k, kp, xp, adt, gamma, beta, L)
torch.mps.synchronize()
iters = 10
t0 = time.time()
for _ in range(iters):
    ssd(x, q, k, kp, xp, adt, gamma, beta, L)
torch.mps.synchronize()
print(f"torch MPS ssd fwd ms/call: {(time.time() - t0) / iters * 1000:.2f}")

import numpy as np
import torch

from zenith.mamba3 import ssd

B, T, H, P, N = 4, 512, 20, 64, 128
NX, NS, NH = B * T * H * P, B * T * H * N, B * T * H


def gen(n, salt):
    return np.sin((np.arange(n, dtype=np.int64) * 131 + salt * 7919).astype(np.float64) * 0.001)


x = torch.tensor(gen(NX, 1)).reshape(B, T, H, P)
xp = torch.tensor(gen(NX, 2)).reshape(B, T, H, P)
q = torch.tensor(gen(NS, 3)).reshape(B, T, H, N)
k = torch.tensor(gen(NS, 4)).reshape(B, T, H, N)
kp = torch.tensor(gen(NS, 5)).reshape(B, T, H, N)
adt = torch.tensor(-0.05 * (1.0 + gen(NH, 6))).reshape(B, T, H)
gamma = torch.tensor(0.05 * (1.0 + gen(NH, 7))).reshape(B, T, H)
beta = torch.tensor(0.05 * (1.0 + gen(NH, 8))).reshape(B, T, H)

y = ssd(x, q, k, kp, xp, adt, gamma, beta, chunk_size=64)
print("torch checksum:", y.double().sum().item())

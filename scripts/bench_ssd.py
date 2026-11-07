import time

import torch

from zenith.mamba3 import ssd as ssd_new


def ssd_loop(x, q, k, kp, xp, adt, gamma, beta, chunk_size):
    Bsz, T, H, P = x.shape
    N = q.shape[-1]
    nc = T // chunk_size

    def chunks(t):
        return t.transpose(1, 2).reshape(Bsz, H, nc, chunk_size, *t.shape[3:])

    x, q, k, kp, xp, adt, gamma, beta = map(chunks, (x, q, k, kp, xp, adt, gamma, beta))
    c = adt.cumsum(-1)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device).tril()
    state = x.new_zeros(Bsz, H, P, N)
    ys = []
    for i in range(nc):
        ci = c[:, :, i]
        decay = torch.exp((ci.unsqueeze(-1) - ci.unsqueeze(-2)).masked_fill(~causal, float("-inf")))
        qi, ki, kpi, xi, xpi = q[:, :, i], k[:, :, i], kp[:, :, i], x[:, :, i], xp[:, :, i]
        gi, bi = gamma[:, :, i], beta[:, :, i]
        attn = (qi @ ki.transpose(-1, -2)) * gi.unsqueeze(-2)
        attn_prev = (qi @ kpi.transpose(-1, -2)) * bi.unsqueeze(-2)
        y = (decay * attn) @ xi + (decay * attn_prev) @ xpi
        ys.append(y + torch.exp(ci).unsqueeze(-1) * (qi @ state.transpose(-1, -2)))
        w = torch.exp(ci[..., -1:] - ci)
        state = torch.exp(ci[..., -1:]).unsqueeze(-1) * state
        state = state + ((w * gi).unsqueeze(-1) * xi).transpose(-1, -2) @ ki
        state = state + ((w * bi).unsqueeze(-1) * xpi).transpose(-1, -2) @ kpi
    return torch.stack(ys, dim=2).reshape(Bsz, H, T, P).transpose(1, 2)


B, T, H, P, N, L = 4, 512, 20, 64, 128, 64
dev = "mps"
torch.manual_seed(0)
base = [torch.randn(B, T, H, P), torch.randn(B, T, H, P), torch.randn(B, T, H, N), torch.randn(B, T, H, N),
        torch.randn(B, T, H, N)]
adt0 = -0.05 * torch.rand(B, T, H)
g0 = 0.05 * torch.rand(B, T, H)
b0 = 0.05 * torch.rand(B, T, H)


def run(fn, dtype):
    x, xp, q, k, kp = [t.to(dev, dtype).requires_grad_() for t in (base[0], base[1], base[2], base[3], base[4])]
    adt, g, bb = [t.to(dev, dtype).requires_grad_() for t in (adt0, g0, b0)]
    args = (x, q, k, kp, xp, adt, g, bb, L)

    def step():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            out = fn(*args)
        out.float().sum().backward()

    for _ in range(2):
        step()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        step()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / 5 * 1000


compiled = torch.compile(ssd_new)
print(f"compiled batched fwd+bwd (autocast): {run(lambda *a: compiled(*a), torch.float32):6.0f} ms")
print(f"batched fwd+bwd (autocast): {run(ssd_new, torch.float32):6.0f} ms")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def heavy_tail(x):
    return x.clamp(min=0) + torch.reciprocal(1 - x.clamp(max=0))


def rope(x, theta, rope_dim):
    rot, rest = x[..., :rope_dim], x[..., rope_dim:]
    a, b = rot[..., 0::2], rot[..., 1::2]
    cos, sin = torch.cos(theta), torch.sin(theta)
    out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    return torch.cat((out, rest), dim=-1)


def ssd(x, q, k, kp, xp, adt, gamma, beta, chunk_size):
    Bsz, T, H, P = x.shape
    N = q.shape[-1]
    assert T % chunk_size == 0, "sequence length must be a multiple of chunk_size"
    nc = T // chunk_size

    def chunks(t):
        return t.transpose(1, 2).reshape(Bsz, H, nc, chunk_size, *t.shape[3:])

    x, q, k, kp, xp, adt, gamma, beta = map(chunks, (x, q, k, kp, xp, adt, gamma, beta))
    c = adt.cumsum(-1)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device).tril()
    decay = torch.exp((c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~causal, float("-inf")))
    attn = (q @ k.transpose(-1, -2)) * gamma.unsqueeze(-2)
    attn_prev = (q @ kp.transpose(-1, -2)) * beta.unsqueeze(-2)
    y_intra = (decay * attn) @ x + (decay * attn_prev) @ xp
    w = torch.exp(c[..., -1:] - c)
    s_local = ((w * gamma).unsqueeze(-1) * x).transpose(-1, -2) @ k + ((w * beta).unsqueeze(-1) * xp).transpose(-1, -2) @ kp
    chunk_decay = torch.exp(c[..., -1])
    states = []
    state = x.new_zeros(Bsz, H, P, N)
    for i in range(nc):
        states.append(state)
        state = chunk_decay[:, :, i, None, None] * state + s_local[:, :, i]
    s_in = torch.stack(states, dim=2)
    y = y_intra + torch.exp(c).unsqueeze(-1) * (q @ s_in.transpose(-1, -2))
    return y.reshape(Bsz, H, T, P).transpose(1, 2)


class Mamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        chunk_size=64,
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
    ):
        super().__init__()
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.A_floor = A_floor
        self.scan = ssd
        self.d_inner = expand * d_model
        self.nheads = self.d_inner // headdim
        self.num_angles = d_state // 4
        self.rope_dim = 2 * self.num_angles

        n_in = 2 * self.d_inner + 2 * d_state + 3 * self.nheads + self.num_angles
        self.in_proj = nn.Linear(d_model, n_in, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))

    def forward(self, u):
        Bsz, T, _ = u.shape
        H, P, N = self.nheads, self.headdim, self.d_state
        z, x, Bm, Cm, dd_dt, dd_A, trap, ang = torch.split(
            self.in_proj(u), [self.d_inner, self.d_inner, N, N, H, H, H, self.num_angles], dim=-1
        )
        x = x.float().view(Bsz, T, H, P)
        z = z.float().view(Bsz, T, H, P)

        A = -heavy_tail(dd_A.float()).clamp(max=-self.A_floor)
        DT = F.softplus(dd_dt.float() + self.dt_bias)
        ADT = A * DT
        trap = torch.sigmoid(trap.float())
        gamma = trap * DT
        beta = (1 - trap) * DT * torch.exp(ADT)
        theta = torch.cumsum(DT.unsqueeze(-1) * ang.float().unsqueeze(2), dim=1)

        q = rope(self.C_norm(Cm.float()).unsqueeze(2) + self.C_bias, theta, self.rope_dim)
        k = rope(self.B_norm(Bm.float()).unsqueeze(2) + self.B_bias, theta, self.rope_dim)
        xp = F.pad(x, (0, 0, 0, 0, 1, 0))[:, :-1]
        kp = F.pad(k, (0, 0, 0, 0, 1, 0))[:, :-1]

        y = self.scan(x, q, k, kp, xp, ADT, gamma, beta, self.chunk_size)
        y = (y + self.D[:, None] * x) * F.silu(z)
        return self.out_proj(y.reshape(Bsz, T, self.d_inner).to(u.dtype))

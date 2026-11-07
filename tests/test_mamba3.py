import math

import pytest
import torch
import torch.nn.functional as F

from zenith.mamba3 import Mamba3, RMSNorm, heavy_tail, rope, ssd


def naive(x, q, k, kp, xp, adt, gamma, beta):
    Bsz, T, H, P = x.shape
    N = q.shape[-1]
    h = x.new_zeros(Bsz, H, P, N)
    ys = []
    for t in range(T):
        a = torch.exp(adt[:, t])[..., None, None]
        g = gamma[:, t][..., None, None]
        b = beta[:, t][..., None, None]
        h = a * h + g * x[:, t][..., None] * k[:, t][..., None, :] + b * xp[:, t][..., None] * kp[:, t][..., None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", h, q[:, t]))
    return torch.stack(ys, 1)


def test_chunked_ssd_matches_naive_recurrence():
    torch.manual_seed(0)
    Bsz, T, H, P, N, L = 2, 128, 3, 4, 8, 16
    x, xp = torch.randn(2, Bsz, T, H, P, dtype=torch.float64)
    q, k, kp = torch.randn(3, Bsz, T, H, N, dtype=torch.float64)
    adt = -torch.rand(Bsz, T, H, dtype=torch.float64)
    gamma, beta = torch.rand(2, Bsz, T, H, dtype=torch.float64)
    fast = ssd(x, q, k, kp, xp, adt, gamma, beta, L)
    slow = naive(x, q, k, kp, xp, adt, gamma, beta)
    assert torch.allclose(fast, slow, atol=1e-9)


def test_mamba3_is_causal():
    torch.manual_seed(0)
    layer = Mamba3(d_model=64, d_state=16, headdim=16, chunk_size=16)
    u = torch.randn(1, 64, 64)
    base = layer(u)
    u2 = u.clone()
    u2[:, -1] += 1.0
    assert torch.allclose(base[:, :-1], layer(u2)[:, :-1], atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 4, 16, 64])
def test_chunked_ssd_is_independent_of_chunk_size(chunk):
    torch.manual_seed(1)
    Bsz, T, H, P, N = 1, 64, 2, 3, 5
    x, xp = torch.randn(2, Bsz, T, H, P, dtype=torch.float64)
    q, k, kp = torch.randn(3, Bsz, T, H, N, dtype=torch.float64)
    adt = -torch.rand(Bsz, T, H, dtype=torch.float64)
    gamma, beta = torch.rand(2, Bsz, T, H, dtype=torch.float64)
    assert torch.allclose(ssd(x, q, k, kp, xp, adt, gamma, beta, chunk), naive(x, q, k, kp, xp, adt, gamma, beta), atol=1e-9)


def test_ssd_rejects_length_not_multiple_of_chunk():
    t = torch.zeros(1, 10, 1, 2, dtype=torch.float64)
    n = torch.zeros(1, 10, 1, 3, dtype=torch.float64)
    a = torch.zeros(1, 10, 1, dtype=torch.float64)
    with pytest.raises(AssertionError, match="multiple of chunk_size"):
        ssd(t, n, n, n, t, a, a, a, 4)


def test_heavy_tail_matches_piecewise_definition():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert torch.allclose(heavy_tail(x), torch.tensor([1 / 3, 0.5, 1.0, 2.0, 3.0]))


def test_heavy_tail_is_continuous_at_zero():
    eps = 1e-6
    left, right = heavy_tail(torch.tensor([-eps, eps])).tolist()
    assert abs(left - 1.0) < 1e-5 and abs(right - 1.0) < 1e-5


def test_rmsnorm_outputs_unit_rms_with_unit_weight():
    torch.manual_seed(0)
    x = torch.randn(4, 16) * 7
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-4)


def test_rmsnorm_applies_learned_weight():
    norm = RMSNorm(8)
    x = torch.randn(2, 8)
    base = norm(x)
    with torch.no_grad():
        norm.weight.mul_(2.0)
    assert torch.allclose(norm(x), 2 * base)


def test_rope_with_zero_angle_is_identity():
    x = torch.randn(1, 3, 2, 8)
    assert torch.allclose(rope(x, torch.zeros(1, 3, 2, 2), rope_dim=4), x)


def test_rope_preserves_norm_of_rotated_pairs_and_leaves_tail_untouched():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, 8)
    theta = torch.randn(2, 5, 3, 2)
    y = rope(x, theta, rope_dim=4)
    assert torch.allclose(y[..., 4:], x[..., 4:])
    assert torch.allclose(y[..., :4].pow(2).sum(-1), x[..., :4].pow(2).sum(-1), atol=1e-5)


def test_rope_rotates_adjacent_pairs_by_angle():
    x = torch.tensor([[[[1.0, 0.0]]]])
    y = rope(x, torch.tensor([[[[math.pi / 2]]]]), rope_dim=2)
    assert torch.allclose(y, torch.tensor([[[[0.0, 1.0]]]]), atol=1e-6)


def test_mamba3_dimensions_follow_config():
    layer = Mamba3(d_model=64, d_state=16, expand=2, headdim=16)
    assert layer.d_inner == 128 and layer.nheads == 8 and layer.num_angles == 4 and layer.rope_dim == 8
    assert layer.in_proj.out_features == 2 * 128 + 2 * 16 + 3 * 8 + 4
    assert layer.out_proj.in_features == 128 and layer.out_proj.out_features == 64


def test_mamba3_dt_bias_produces_timesteps_in_configured_range():
    torch.manual_seed(0)
    layer = Mamba3(d_model=64, d_state=16, headdim=16, dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4)
    dt = F.softplus(layer.dt_bias)
    assert torch.all(dt >= 1e-4 * 0.99) and torch.all(dt <= 1e-1 * 1.01)


def test_mamba3_output_shape_and_gradient_reaches_every_parameter():
    torch.manual_seed(0)
    layer = Mamba3(d_model=32, d_state=16, headdim=16, chunk_size=8)
    u = torch.randn(2, 16, 32, requires_grad=True)
    out = layer(u)
    assert out.shape == u.shape
    out.square().mean().backward()
    missing = [n for n, p in layer.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert missing == []


def test_mamba3_forward_uses_injected_scan():
    layer = Mamba3(d_model=32, d_state=16, expand=2, headdim=8, chunk_size=8)
    calls = []

    def spy(*args):
        calls.append(args[-1])
        return ssd(*args)

    layer.scan = spy
    u = torch.randn(1, 16, 32)
    assert torch.allclose(layer(u), Mamba3.forward(layer, u))
    assert calls == [8, 8]

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

import zenith.mamba3 as mamba3
from zenith.model import ZenithConfig, ZenithForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/target/config.json")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--loops", type=int, default=2)
parser.add_argument("--iters", type=int, default=5)
parser.add_argument("--compile_scan", action="store_true")
args = parser.parse_args()

dev = "mps"
torch.manual_seed(0)
cfg = ZenithConfig.from_json_file(args.config)
model = ZenithForCausalLM(cfg).to(dev)
if args.compile_scan:
    compiled_scan = torch.compile(mamba3.ssd)
    for m in model.modules():
        if isinstance(m, mamba3.Mamba3):
            m.scan = compiled_scan
ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=dev)
times = defaultdict(float)


@contextmanager
def timed(name):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    yield
    torch.mps.synchronize()
    times[name] += time.perf_counter() - t0


_ssd = mamba3.ssd
_mixer_fwd = mamba3.Mamba3.forward
_lm = model.lm_head


def ssd_timed(*a, **kw):
    with timed("scan (ssd)"):
        return _ssd(*a, **kw)


def mixer_timed(self, u):
    with timed("mixer total"):
        return _mixer_fwd(self, u)


class LMTimed(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.weight = inner.weight

    def forward(self, x):
        with timed("lm_head"):
            return self.inner(x)


mamba3.ssd = ssd_timed
mamba3.Mamba3.forward = mixer_timed
model.lm_head = LMTimed(_lm)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)


def step(loops):
    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        out = model(ids, labels=ids, loops=loops)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


for _ in range(2):
    step(args.loops)
times.clear()
torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(args.iters):
    step(args.loops)
torch.mps.synchronize()
total = (time.perf_counter() - t0) / args.iters
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  batch {args.batch_size}x{args.seq_len}  loops {args.loops}")
print(f"full step (fwd+bwd+opt): {total*1000:.0f} ms")
for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
    print(f"  forward {k:14s}: {v/args.iters*1000:7.0f} ms per step")

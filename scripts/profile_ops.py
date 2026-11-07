import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from zenith.model import ZenithConfig, ZenithForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/target/config.json")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--loops", type=int, default=2)
parser.add_argument("--top", type=int, default=25)
args = parser.parse_args()

dev = "mps"
torch.manual_seed(0)
cfg = ZenithConfig.from_json_file(args.config)
model = ZenithForCausalLM(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=dev)


def step():
    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        out = model(ids, labels=ids, loops=args.loops)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


for _ in range(2):
    step()
torch.mps.synchronize()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    step()
    torch.mps.synchronize()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=args.top))

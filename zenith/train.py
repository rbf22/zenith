import argparse
import glob
import json
import math
import os
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from zenith import ZenithConfig, ZenithForCausalLM
from zenith.mamba3 import Mamba3, ssd
from zenith.loop import sample_loops


def load_tokens(patterns, tokenizer):
    texts = [json.loads(line)["text"] for pattern in patterns for path in sorted(glob.glob(pattern)) for line in open(path)]
    ids = []
    for i in range(0, len(texts), 1000):
        for enc in tokenizer(texts[i : i + 1000])["input_ids"]:
            ids.extend(enc + [tokenizer.eos_token_id])
    return torch.tensor(ids, dtype=torch.long)


def get_batch(tokens, batch_size, seq_len, gen):
    starts = torch.randint(0, len(tokens) - seq_len, (batch_size,), generator=gen)
    return torch.stack([tokens[s : s + seq_len] for s in starts])


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def amp_context(device):
    use_bf16 = device == "mps" or (device == "cuda" and torch.cuda.is_bf16_supported())
    return torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16)


def cosine_lambda(step, warmup, total, min_ratio=0.1):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", nargs="+", required=True, help="glob(s) of JSONL files with a 'text' key")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    p.add_argument("--compile_scan", action="store_true", help="torch.compile the SSD scan (about 2x faster on MPS)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    cfg = ZenithConfig(**json.load(open(args.config)))
    state_path = os.path.join(args.output_dir, "train_state.pt")
    resume = os.path.exists(state_path)
    model = (ZenithForCausalLM.from_pretrained(args.output_dir) if resume else ZenithForCausalLM(cfg)).to(device)
    if args.compile_scan:
        scan = torch.compile(ssd)
        for m in model.modules():
            if isinstance(m, Mamba3):
                m.scan = scan
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M on {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = load_tokens(args.data, tokenizer)
    print(f"tokens: {len(tokens):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    sched = LambdaLR(opt, partial(cosine_lambda, warmup=args.warmup_steps, total=args.max_steps))
    gen = torch.Generator().manual_seed(args.seed)
    start = 0
    if resume:
        state = torch.load(state_path, weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        gen.set_state(state["gen"])
        start = state["step"]
        print(f"resumed from step {start}", flush=True)

    model.train()
    for step in range(start, args.max_steps):
        k = sample_loops(step, 0, max_loops=cfg.max_loops)
        losses = []
        for _ in range(args.grad_accum):
            ids = get_batch(tokens, args.batch_size, args.seq_len, gen).to(device)
            with amp_context(device):
                loss = model(ids, labels=ids, loops=k).loss
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_steps == 0:
            print(json.dumps({"step": step, "loss": round(sum(losses) / len(losses), 4), "loops": k, "lr": sched.get_last_lr()[0]}), flush=True)
        if (step + 1) % args.save_steps == 0 or step + 1 == args.max_steps:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            tmp = state_path + ".tmp"
            torch.save({"step": step + 1, "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": gen.get_state()}, tmp)
            os.replace(tmp, state_path)


if __name__ == "__main__":
    main()

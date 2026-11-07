import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zenith import ZenithConfig, ZenithForCausalLM
from zenith.train import amp_context, pick_device


def windows(tokenizer, path, seq_len, max_windows):
    texts = [json.loads(line)["text"] for line in open(path)]
    ids = []
    for enc in tokenizer(texts)["input_ids"]:
        ids.extend(enc + [tokenizer.eos_token_id])
    n = min(len(ids) // seq_len, max_windows)
    return torch.tensor(ids[: n * seq_len]).view(n, seq_len)


def evaluate(model, data, loops, batch_size, device):
    total, count = 0.0, 0
    with torch.no_grad(), amp_context(device):
        for i in range(0, len(data), batch_size):
            ids = data[i : i + batch_size].to(device)
            loss = model(ids, labels=ids, loops=loops).loss
            total += loss.item() * ids.numel()
            count += ids.numel()
    return total / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="model dirs; 'random' for an untrained model")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_windows", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--loops", nargs="+", type=int, default=[1, 4])
    p.add_argument("--random_config", default="configs/target/config.json")
    args = p.parse_args()

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data = windows(tokenizer, args.heldout, args.seq_len, args.max_windows)
    print(f"held-out windows: {len(data)} x {args.seq_len} tokens", flush=True)
    for ckpt in args.checkpoints:
        if ckpt == "random":
            torch.manual_seed(0)
            model = ZenithForCausalLM(ZenithConfig(**json.load(open(args.random_config)))).to(device).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt).to(device).eval()
        results = {loop: round(evaluate(model, data, loop, args.batch_size, device), 4) for loop in args.loops}
        print(json.dumps({"checkpoint": ckpt, "heldout_loss": results}), flush=True)

if __name__ == "__main__":
    main()

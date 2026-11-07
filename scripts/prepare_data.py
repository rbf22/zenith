import argparse
import json
import os
import random

from datasets import load_dataset

SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, 63.75),
    "open-web-math": ("open-web-math/open-web-math", {}, 6.93),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_chars", type=int, required=True, help="total characters for training data")
    p.add_argument("--tokenizer_chars", type=int, required=True, help="characters sampled for tokenizer training")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = list(SOURCES)
    weights = [SOURCES[n][2] for n in names]
    streams = {n: iter(load_dataset(SOURCES[n][0], split="train", streaming=True, **SOURCES[n][1])) for n in names}
    rng = random.Random(args.seed)

    written = {n: 0 for n in names}
    total = 0
    tok_path = os.path.join(args.out_dir, "tokenizer_sample.jsonl")
    train_path = os.path.join(args.out_dir, "train.jsonl")
    with open(train_path, "w") as train_f, open(tok_path, "w") as tok_f:
        while total < args.train_chars:
            name = rng.choices(names, weights)[0]
            text = next(streams[name])["text"]
            if total < args.tokenizer_chars:
                tok_f.write(json.dumps({"text": text}) + "\n")
            train_f.write(json.dumps({"text": text}) + "\n")
            total += len(text)
            written[name] += 1
    print(json.dumps({"chars": total, "docs": written}))


if __name__ == "__main__":
    main()

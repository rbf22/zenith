import argparse
import json

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--docs", type=int, default=400)
    p.add_argument("--skip", type=int, default=300000, help="skip this many stream docs to stay clear of training data")
    args = p.parse_args()
    stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")
    n = 0
    with open(args.out, "w") as f:
        for row in stream.skip(args.skip):
            f.write(json.dumps({"text": row["text"]}) + "\n")
            n += 1
            if n >= args.docs:
                break
    print(json.dumps({"docs": n, "out": args.out}))


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, log_path=None):
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    with open(log_path, "w") as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive", required=True, help="folder for data and results, e.g. /content/drive/MyDrive/zenith")
    p.add_argument("--name", required=True, help="run name; results go to <drive>/results/<name>")
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    args = p.parse_args()

    data = os.path.join(args.drive, "zenith_data")
    tok = os.path.join(ROOT, "tokenizer32k")
    out = os.path.join(args.drive, "results", args.name)
    os.makedirs(out, exist_ok=True)

    if not os.path.exists(os.path.join(data, "train.jsonl")):
        os.makedirs(data, exist_ok=True)
        run([sys.executable, "scripts/prepare_data.py", "--out_dir", data, "--train_chars", "150001554", "--tokenizer_chars", "100000000"])
    if not os.path.exists(os.path.join(data, "heldout.jsonl")):
        run([sys.executable, "scripts/make_heldout.py", "--out", os.path.join(data, "heldout.jsonl"), "--docs", "400"])

    ckpt = os.path.join(out, "ckpt")
    if not os.path.exists(os.path.join(ckpt, "model.safetensors")):
        run([sys.executable, "-m", "zenith.train", "--config", "configs/target/config.json", "--tokenizer", tok,
             "--data", os.path.join(data, "train.jsonl"), "--output_dir", ckpt, "--seq_len", "512", "--batch_size", "2",
             "--max_steps", str(args.steps), "--warmup_steps", str(args.warmup), "--lr", args.lr,
             "--save_steps", "100000", "--log_steps", "25", "--compile_scan"], log_path=os.path.join(out, "train.log"))

    run([sys.executable, "scripts/eval_ckpt.py", "--checkpoints", ckpt, "--tokenizer", tok,
         "--heldout", os.path.join(data, "heldout.jsonl"), "--max_windows", "128"], log_path=os.path.join(out, "eval.log"))
    print(open(os.path.join(out, "eval.log")).read().strip().splitlines()[-1])


if __name__ == "__main__":
    main()

import json
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM

from zenith import train
from zenith.train import cosine_lambda, get_batch, load_tokens


def write_jsonl(path, texts):
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return path


def test_cosine_lambda_warms_up_linearly():
    assert cosine_lambda(0, warmup=10, total=100) == 0.0
    assert cosine_lambda(5, warmup=10, total=100) == pytest.approx(0.5)


def test_cosine_lambda_decays_to_min_ratio_at_end():
    assert cosine_lambda(10, warmup=10, total=100) == pytest.approx(1.0)
    assert cosine_lambda(100, warmup=10, total=100) == pytest.approx(0.1)
    assert cosine_lambda(500, warmup=10, total=100) == pytest.approx(0.1)


def test_cosine_lambda_is_midpoint_halfway_through_decay():
    assert cosine_lambda(55, warmup=10, total=100) == pytest.approx(0.55)


def test_cosine_lambda_handles_zero_warmup():
    assert cosine_lambda(0, warmup=0, total=10) == pytest.approx(1.0)


def test_get_batch_returns_contiguous_windows():
    tokens = torch.arange(1000)
    batch = get_batch(tokens, batch_size=3, seq_len=8, gen=torch.Generator().manual_seed(0))
    assert batch.shape == (3, 8)
    for row in batch:
        assert torch.equal(row, torch.arange(row[0], row[0] + 8))


def test_get_batch_is_reproducible_with_same_seed():
    tokens = torch.arange(1000)
    a = get_batch(tokens, 4, 16, torch.Generator().manual_seed(3))
    b = get_batch(tokens, 4, 16, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_load_tokens_appends_eos_after_each_document(tmp_path, tokenizer):
    write_jsonl(tmp_path / "a.jsonl", ["the quick fox", "lazy dog"])
    tokens = load_tokens([str(tmp_path / "*.jsonl")], tokenizer)
    expected = []
    for text in ["the quick fox", "lazy dog"]:
        expected += tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
    assert tokens.tolist() == expected


def test_load_tokens_reads_every_matching_file_and_pattern(tmp_path, tokenizer):
    write_jsonl(tmp_path / "x1.jsonl", ["alpha"])
    write_jsonl(tmp_path / "x2.jsonl", ["beta"])
    write_jsonl(tmp_path / "y.jsonl", ["gamma"])
    one = load_tokens([str(tmp_path / "x*.jsonl")], tokenizer)
    both = load_tokens([str(tmp_path / "x*.jsonl"), str(tmp_path / "y.jsonl")], tokenizer)
    assert len(both) > len(one)


def test_load_tokens_handles_more_than_one_batch_of_documents(tmp_path, tokenizer):
    write_jsonl(tmp_path / "big.jsonl", ["the quick brown fox"] * 1500)
    tokens = load_tokens([str(tmp_path / "big.jsonl")], tokenizer)
    per_doc = len(tokenizer("the quick brown fox")["input_ids"]) + 1
    assert len(tokens) == 1500 * per_doc


def test_cli_entry_point_lists_options():
    result = subprocess.run([sys.executable, "-m", "zenith.train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout and "--max_steps" in result.stdout


def test_main_trains_saves_and_reloads(tmp_path, tokenizer, tiny_config_path, monkeypatch):
    data = write_jsonl(tmp_path / "train.jsonl", ["the quick brown fox jumps over the lazy dog"] * 40)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train", "--config", str(tiny_config_path), "--tokenizer", str(tmp_path / "tokenizer"),
        "--data", str(data), "--output_dir", str(out), "--seq_len", "32", "--batch_size", "2",
        "--max_steps", "3", "--warmup_steps", "1", "--save_steps", "2", "--log_steps", "1",
    ])
    train.main()
    assert (out / "config.json").exists()
    assert AutoModelForCausalLM.from_pretrained(out) is not None
    assert (out / "tokenizer_config.json").exists() or (out / "tokenizer.json").exists()


def test_main_with_compile_scan_wraps_every_mamba_scan(tmp_path, tokenizer, tiny_config_path, monkeypatch):
    data = write_jsonl(tmp_path / "train.jsonl", ["the quick brown fox jumps over the lazy dog"] * 40)
    compiled = []

    def fake_compile(fn):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(train.torch, "compile", fake_compile)
    monkeypatch.setattr(sys, "argv", [
        "train", "--config", str(tiny_config_path), "--tokenizer", str(tmp_path / "tokenizer"),
        "--data", str(data), "--output_dir", str(tmp_path / "run"), "--seq_len", "32", "--batch_size", "2",
        "--max_steps", "1", "--warmup_steps", "1", "--save_steps", "10", "--log_steps", "1", "--compile_scan",
    ])
    train.main()
    assert compiled == [train.ssd]


def test_main_resumes_optimizer_schedule_and_data_position(tmp_path, tokenizer, tiny_config_path, monkeypatch, capsys):
    data = write_jsonl(tmp_path / "train.jsonl", ["the quick brown fox jumps over the lazy dog"] * 40)
    out = tmp_path / "run"

    def run(max_steps):
        monkeypatch.setattr(sys, "argv", [
            "train", "--config", str(tiny_config_path), "--tokenizer", str(tmp_path / "tokenizer"),
            "--data", str(data), "--output_dir", str(out), "--seq_len", "32", "--batch_size", "2",
            "--max_steps", str(max_steps), "--warmup_steps", "1", "--save_steps", "100", "--log_steps", "1",
        ])
        train.main()

    run(2)
    assert (out / "train_state.pt").exists()
    capsys.readouterr()
    run(4)
    output = capsys.readouterr().out
    assert "resumed from step 2" in output
    assert [json.loads(line)["step"] for line in output.splitlines() if line.startswith('{"step"')] == [2, 3]


def test_pick_device_matches_available_accelerator():
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert train.pick_device() == expected


def test_amp_context_is_disabled_on_cpu():
    with train.amp_context("cpu"):
        assert not torch.is_autocast_enabled("cpu")


def test_pick_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    assert train.pick_device() == "cuda"


def test_main_grad_accum_draws_one_batch_per_micro_step(tmp_path, tokenizer, tiny_config_path, monkeypatch):
    data = write_jsonl(tmp_path / "train.jsonl", ["the quick brown fox jumps over the lazy dog"] * 40)
    calls = []
    real_get_batch = train.get_batch

    def counting_get_batch(*args):
        calls.append(1)
        return real_get_batch(*args)

    monkeypatch.setattr(train, "get_batch", counting_get_batch)
    monkeypatch.setattr(sys, "argv", [
        "train", "--config", str(tiny_config_path), "--tokenizer", str(tmp_path / "tokenizer"),
        "--data", str(data), "--output_dir", str(tmp_path / "run"), "--seq_len", "32", "--batch_size", "2",
        "--max_steps", "2", "--warmup_steps", "1", "--save_steps", "10", "--log_steps", "1", "--grad_accum", "2",
    ])
    train.main()
    assert len(calls) == 4

import pathlib

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture
def tiny_config_path():
    return ROOT / "configs" / "tiny" / "config.json"


@pytest.fixture
def tokenizer(tmp_path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<eos>"]))
    out = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<eos>")
    out.save_pretrained(tmp_path / "tokenizer")
    return out

import argparse
import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

EOS = "[EOS]"


def texts(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)["text"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, help="JSONL with a 'text' key")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vocab_size", type=int, default=32000)
    args = p.parse_args()

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=[EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts(args.sample), trainer)
    out = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, pad_token=EOS)
    out.save_pretrained(args.out_dir)
    print(json.dumps({"vocab_size": out.vocab_size, "eos_id": out.eos_token_id}))


if __name__ == "__main__":
    main()

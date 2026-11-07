from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .mamba3 import Mamba3, RMSNorm


class ZenithConfig(PretrainedConfig):
    model_type = "zenith"

    def __init__(
        self,
        vocab_size=32000,
        d_model=640,
        d_state=128,
        expand=2,
        headdim=64,
        chunk_size=64,
        n_input=4,
        n_recurrent=16,
        n_output=4,
        max_loops=4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.n_input = n_input
        self.n_recurrent = n_recurrent
        self.n_output = n_output
        self.max_loops = max_loops
        super().__init__(**kwargs)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = Mamba3(cfg.d_model, cfg.d_state, cfg.expand, cfg.headdim, cfg.chunk_size)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class ZenithForCausalLM(PreTrainedModel):
    config_class = ZenithConfig
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.input_layers = nn.ModuleList(Layer(config) for _ in range(config.n_input))
        self.recurrent_layers = nn.ModuleList(Layer(config) for _ in range(config.n_recurrent))
        self.output_layers = nn.ModuleList(Layer(config) for _ in range(config.n_output))
        self.recurrent_norm = RMSNorm(config.d_model)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, labels=None, loops=None, **kwargs):
        loops = loops or self.config.max_loops
        x = self.embed(input_ids)
        for layer in self.input_layers:
            x = layer(x)
        s = x
        for t in range(loops):
            with torch.no_grad() if t < loops - 1 else nullcontext():
                h = self.recurrent_norm(s + x)
                for layer in self.recurrent_layers:
                    h = layer(h)
                s = h
        for layer in self.output_layers:
            s = layer(s)
        logits = self.lm_head(self.norm(s))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1)
            )
        return CausalLMOutput(loss=loss, logits=logits)

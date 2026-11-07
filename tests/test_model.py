import json
import pathlib

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from zenith import ZenithConfig, ZenithForCausalLM
from zenith.model import Layer

TINY = pathlib.Path(__file__).parent.parent / "configs" / "tiny" / "config.json"


def tiny_model():
    torch.manual_seed(0)
    return ZenithForCausalLM(ZenithConfig(**json.loads(TINY.read_text())))


def test_forward_and_loss_are_finite_for_each_loop_count():
    model = tiny_model()
    ids = torch.randint(0, 1024, (2, 64))
    for k in range(1, 5):
        out = model(ids, labels=ids, loops=k)
        assert out.logits.shape == (2, 64, 1024)
        assert torch.isfinite(out.loss)


def test_more_loops_change_output():
    model = tiny_model()
    ids = torch.randint(0, 1024, (1, 64))
    assert not torch.allclose(model(ids, loops=1).logits, model(ids, loops=2).logits)


def test_only_last_loop_is_tracked_by_autograd():
    model = tiny_model()
    seen = {}
    for i, layer in enumerate(model.recurrent_layers):
        layer.register_forward_hook(lambda m, inp, out, i=i: seen.setdefault(i, []).append(out.requires_grad))
    ids = torch.randint(0, 1024, (1, 64))
    model(ids, labels=ids, loops=3).loss.backward()
    assert all(flags == [False, False, True] for flags in seen.values())


def test_target_config_is_under_200m_params():
    cfg = ZenithConfig(**json.loads((TINY.parent.parent / "target" / "config.json").read_text()))
    with torch.device("meta"):
        model = ZenithForCausalLM(cfg)
    assert sum(p.numel() for p in model.parameters()) < 200_000_000


def test_loss_is_none_without_labels():
    model = tiny_model()
    assert model(torch.randint(0, 1024, (1, 32))).loss is None


def test_loss_matches_manual_next_token_cross_entropy():
    model = tiny_model()
    ids = torch.randint(0, 1024, (2, 32))
    out = model(ids, labels=ids, loops=2)
    expected = F.cross_entropy(out.logits[:, :-1].reshape(-1, 1024), ids[:, 1:].reshape(-1))
    assert torch.allclose(out.loss, expected, atol=1e-5)


def test_default_loops_is_config_max_loops():
    model = tiny_model()
    ids = torch.randint(0, 1024, (1, 32))
    assert torch.allclose(model(ids).logits, model(ids, loops=model.config.max_loops).logits)


def test_lm_head_is_tied_to_embedding():
    model = tiny_model()
    assert model.lm_head.weight is model.embed.weight


def test_embedding_accessors_return_and_replace_modules():
    model = tiny_model()
    assert model.get_input_embeddings() is model.embed
    assert model.get_output_embeddings() is model.lm_head
    new_embed = torch.nn.Embedding(1024, model.config.d_model)
    model.set_input_embeddings(new_embed)
    assert model.embed is new_embed
    new_head = torch.nn.Linear(model.config.d_model, 1024, bias=False)
    model.set_output_embeddings(new_head)
    assert model.lm_head is new_head


def test_layer_is_identity_when_mixer_output_is_zero():
    layer = Layer(tiny_model().config)
    torch.nn.init.zeros_(layer.mixer.out_proj.weight)
    x = torch.randn(1, 32, layer.norm.weight.numel())
    assert torch.allclose(layer(x), x)


def test_save_and_reload_round_trips_weights(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert type(reloaded) is ZenithForCausalLM
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), name


def test_auto_classes_resolve_zenith_config_and_model():
    assert AutoConfig.for_model("zenith").__class__ is ZenithConfig
    assert AutoModelForCausalLM._model_mapping[ZenithConfig] is ZenithForCausalLM

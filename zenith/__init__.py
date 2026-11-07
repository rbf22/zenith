from transformers import AutoConfig, AutoModelForCausalLM

from .model import ZenithConfig, ZenithForCausalLM

AutoConfig.register("zenith", ZenithConfig)
AutoModelForCausalLM.register(ZenithConfig, ZenithForCausalLM)

__all__ = ["ZenithConfig", "ZenithForCausalLM"]

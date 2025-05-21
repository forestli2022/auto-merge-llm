import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoModel

from .custom_llama import CustomLlamaForCausalLM
from .custom_llama_vllm import CustomLlamaForCausalLM as CustomLlamaModelVllm
from .custom_llama_config import CustomLlamaConfig
from .custom_filter import GetAnswer
from .custom_filter import GetCode
from .custom_filter import GetAnswerZh
from .custom_filter import GetAnswerJa


AutoConfig.register("customllama", CustomLlamaConfig)
AutoModelForCausalLM.register(CustomLlamaConfig, CustomLlamaForCausalLM)
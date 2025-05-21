from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn as nn

from .custom_llama_config import CustomLlamaConfig


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, scale=1.0):
        super().__init__(config, layer_idx)
        self.scale = scale

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=True):
        outputs = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        hidden_states = outputs[0]
        hidden_states = hidden_states * self.scale
        return (hidden_states, *outputs[1:])


class CustomLlama(LlamaModel):
    config_class = CustomLlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.scales = config.scales
        assert len(self.scales) == config.num_hidden_layers

        self.layers = nn.ModuleList([
            CustomLlamaDecoderLayer(config, layer_idx=i,scale=self.scales[i]) for i in range(config.num_hidden_layers)
        ])

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class CustomLlamaForCausalLM(LlamaForCausalLM):
    config_class = CustomLlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlama(config)
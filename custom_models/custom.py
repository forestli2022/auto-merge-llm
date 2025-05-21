from typing import List, Optional, Union, Iterable, Tuple
import torch

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaModel, make_layers, LlamaForCausalLM


def create_layer_fn(config, cache_config, quant_config):
    def layer_fn():
        nonlocal layer_idx
        layer = CustomLlamaDecoderLayer(
            config=config,
            layer_idx=layer_idx, 
            cache_config=cache_config,
            quant_config=quant_config,
        )
        layer_idx += 1
        return layer

    layer_idx = 0
    return layer_fn


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: Optional["CacheConfig"] = None,
        quant_config: Optional["QuantizationConfig"] = None,
    ) -> None:
        #print(quant_config)
        super().__init__(config, cache_config, quant_config)
        self.scale = config.scales[layer_idx] if layer_idx < len(config.scales) else 1.0

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = super().forward(
            positions, hidden_states, kv_cache, attn_metadata, residual
        )
        
        hidden_states.mul_(self.scale)
        return hidden_states, residual

class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        
        layer_fn = create_layer_fn(config, cache_config, quant_config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            layer_fn,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
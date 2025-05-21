import os
import safetensors
import torch

from utils import logger
from .load_helper import get_shards_from_disk, LazyPickleLoader


class TensorLoader:
    def __init__(self, model_name, lazy_unpickle,
                 device=None, is_safetensors=False):
        self.model_name = model_name
        self.device = device
        self.is_safetensors = is_safetensors
        self.base_path = None
        self.tensor_paths = None
        self.shards = None
        self.current_shard = None
        self.use_lazy_unpickle = lazy_unpickle
        self.get_shard_tensors()

    def get_tensor(self, key):
        if self.current_shard is None or key not in self.current_shard.keys():
            key = key.removeprefix("model.") if key not in self.tensor_paths else key
            if key not in self.tensor_paths:
                key = "embed_tokens.weight" if key == "lm_head.weight" else key
                if key not in self.tensor_paths:
                    raise KeyError(f"Key '{key}' not found.")

            self.current_shard = None
            self.current_keys = None

            shard_file = self.tensor_paths[key]
            shard_full_path = os.path.join(self.base_path, shard_file)
            logger.debug(f"Opening shard {shard_full_path}")
            self.current_shard = self.get_current_shard_tensors(
                shard_full_path
            )
        if self.is_safetensors or self.use_lazy_unpickle:
            return self.current_shard.get_tensor(key).to(self.device)
        return self.current_shard[key]
               
    def get_current_shard_tensors(self, shard_full_path):
        if shard_full_path.lower().endswith(".safetensors"):
            # not a subclass of TensorLoader, but exposes same api
            return safetensors.safe_open(
                shard_full_path, framework="pt", device=self.device or "cpu"
            )
        elif self.use_lazy_unpickle:
            return LazyPickleLoader(shard_full_path, device=self.device)
        return torch.load(shard_full_path, map_location=self.device, 
                          weights_only=True)
        
    def release(self):
        self.current_shard_tensors = None
    
    def get_shard_tensors(self):
        # Todo: reshard tensors ? 
        self.is_safetensors, self.tensor_paths, self.shards, self.base_path = (
            get_shards_from_disk(self.model_name)
        )
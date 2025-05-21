import os
import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from tokenizer import align_tokenizers_and_embeddings
from utils import get_model_storage_path, logger

CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"))


class MergeMethod(ABC):
    def __init__(self):
        pass  
    
    def prepare_merge(
        self,
        base_model,
        models_to_merge,
        exclude_param_names_regex
    ):
        base_model_dict, merging_model_list = self._load_checkpoints(base_model, models_to_merge)
        align_tokenizers_and_embeddings(
            pretrained_model=base_model_dict['model'], 
            pretrained_tokenizer=base_model_dict['tokenizer'],
            pretrained_config=base_model_dict['config'], 
            finetuned_models=[merging_model['model'] for merging_model in merging_model_list],
            finetuned_tokenizers=[merging_model['tokenizer'] for merging_model in merging_model_list], 
            finetuned_configs=[merging_model['config'] for merging_model in merging_model_list]
        )
        return base_model_dict, merging_model_list
    
    def finalize_merge(
        self,
        base_model,
        base_model_dict,
        merging_model_list,
        averaged_params
    ):
        self.copy_params_to_model(params=averaged_params, model=base_model)
        merged_res = {
            'merged_model': base_model,
            'base_tokenizer': base_model_dict['tokenizer'],
            'merged_model_tokenizers': [merging_model['tokenizer']
                                        for merging_model
                                        in merging_model_list]
        }
        return merged_res
    
    def copy_params_to_model(
        self,
        params,
        model
    ):
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])
    
        
    def _load_checkpoint(
        self,
        model_path
    ):
        res = {}
        try:
            temp_model_path = get_model_storage_path(model_path)
            res['model'] = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=temp_model_path,
                device_map="cpu"
            )
            res['tokenizer'] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=temp_model_path
            )
            res['config'] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=temp_model_path
            )
        except Exception as e:
            logger.error(e)
            res['model'] = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path, 
                cache_dir=CACHE_DIR, 
                device_map="cpu"
            )
            res['tokenizer'] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_path, 
                cache_dir=CACHE_DIR
            )
            res['config'] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_path, 
                cache_dir=CACHE_DIR
            )
        return res
            
    def _load_checkpoints(
        self,
        base_model_path,
        models_to_merge_paths
    ):
        based_model = {}
        merging_model_list = []
        based_model = self._load_checkpoint(base_model_path)
        for model_merge_path in models_to_merge_paths:
            merging_model_list.append(
                self._load_checkpoint(model_merge_path)
            )
        return based_model, merging_model_list
                   
    @abstractmethod
    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        pass
    
    @abstractmethod
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        pass

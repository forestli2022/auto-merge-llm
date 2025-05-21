from collections import defaultdict

import torch

from utils import get_param_names_to_merge, logger
from .base_merge_method import MergeMethod


class AverageMerging(MergeMethod):
    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, 
            exclude_param_names_regex
        ) 
        self.mask_params(
            base_model_dict['model'],
            [model_to_merge['model'] for model_to_merge in models_to_merge],
            exclude_param_names_regex,
            mask_merging
        )
        models_to_merge_param_dict = defaultdict(list)
        base_model = base_model_dict['model']
        # iterate each individual model that needs to be merged
        for model_to_merge_dict in merging_model_list:
            model_to_merge = model_to_merge_dict['model']
            param_dict = {
                param_name: param_value
                for param_name, param_value
                in model_to_merge.named_parameters()
            }
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex
            )
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(
                    param_dict[param_name]
                )  
        with torch.no_grad():
            # average merging of individual models' parameters
            averaged_params = {
                param_name: torch.stack(
                    model_to_merge_param, dim=0
                ).mean(dim=0) 
                for param_name, model_to_merge_param 
                in models_to_merge_param_dict.items()
            }
        
        return self.finalize_merge(
            base_model, 
            base_model_dict,
            merging_model_list, 
            averaged_params
        )
    
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        merged_tensor = torch.stack(
            [
                merging_tensor 
                for merging_tensor in tensors_to_merge
            ],
            dim=0
        ).mean(dim=0)
        return merged_tensor
        
    
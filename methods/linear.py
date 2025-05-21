from collections import defaultdict

import torch

from utils import get_param_names_to_merge, logger
from .base_method import MergeMethod


class LinearMerging(MergeMethod):
    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        weights = method_params["weights"]
        if weights is None:
            weights = [1.0] * len(models_to_merge)  # Default to equal weights
        else:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
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
            merged_params = {}
            for param_name, model_to_merge_param in models_to_merge_param_dict.items():
                # Compute the weighted sum instead of a mean
                weighted_sum = sum(
                    weight * param for weight, param in zip(weights, model_to_merge_param)
                )
                merged_params[param_name] = weighted_sum
        
        return self.finalize_merge(
            base_model, 
            base_model_dict,
            merging_model_list, 
            merged_params
        )
    
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        # Ensure weights are provided and normalized
        weights = method_params["weights"]
        normalize = True #method_params["normalize"]
        if weights is None:
            weights = [1.0] * len(tensors_to_merge)  # Default to equal weights
        if normalize:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]  # Normalize weights
        logger.info(f"linear merge: current merge weights is {weights}")
        merged_tensor = sum(
            weight * tensor.to("cpu") 
            for weight, tensor in zip(weights, tensors_to_merge)
        )
        return merged_tensor
        
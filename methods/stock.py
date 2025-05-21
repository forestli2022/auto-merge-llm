import torch
import torch.nn as nn

from utils import get_param_names_to_merge
from .base_method import MergeMethod


class StockMerging(MergeMethod):
    def merge(
        self, 
        base_model, 
        models_to_merge, 
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        """
        stock merging method
        w = t \cdot w_{avg}^N + (1 - t) \cdot w_0
        t = \frac{N cos(\theta)}{1 + (N - 1) \cdot cos(\theta)}
        """
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, 
            models_to_merge, 
            exclude_param_names_regex
        )
        finetuned_param_dict_list = [
            {
                param_name: param_value 
                for param_name, param_value in model_to_merge['model'].named_parameters()
            } 
            for model_to_merge in merging_model_list
        ]
        pretrained_param_dict = {
            param_name: param_value 
            for param_name, param_value in base_model_dict['model'].named_parameters()
        }
        base_model = base_model_dict['model']
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(
            input_param_names=list(finetuned_param_dict_list[0].keys()), 
            exclude_param_names_regex=exclude_param_names_regex
        )

        # dict, dictionary of merged model parameters
        merged_params = {}
        with torch.no_grad():
            for param_name in param_names_to_merge:
                # Tensor, shape (param_shape, )
                pretrained_param = pretrained_param_dict[param_name]
                # list, each element is a Tensor with shape (param_shape, )
                finetuned_params = [
                    finetuned_param_dict[param_name] 
                    for finetuned_param_dict in finetuned_param_dict_list
                ]

                out_shape = pretrained_param.shape
                pretrained_param = pretrained_param.view(-1)
                finetuned_params = [
                    finetuned_param.view(-1) 
                    for finetuned_param in finetuned_params
                ]

                # follow the mergekit to implement stock merging
                delta_params = [
                    finetuned_param - pretrained_param 
                    for finetuned_param in finetuned_params
                ]

                cos_thetas = []
                for i, delta_param_i in enumerate(delta_params):
                    for j in range(i + 1, len(delta_params)):
                        delta_param_j = delta_params[j]

                        norm_product = (
                            torch.norm(delta_param_i, dim=-1) 
                            * torch.norm(delta_param_j, dim=-1)
                        )
                        cos_theta = (
                            (delta_param_i * delta_param_j).sum(dim=-1)
                            / norm_product.clamp(min=1e-6)
                        ).clamp(min=-1, max=1)
                        cos_thetas.append(cos_theta)

                # Tensor, shape (1, )
                cos_theta = torch.stack(
                    cos_thetas, dim=0
                ).mean(dim=0).unsqueeze(dim=-1)
                N = len(finetuned_params)
                # Tensor, shape (1, )
                t = (N * cos_theta) / (1 + (N - 1) * cos_theta)

                param_avg = sum(finetuned_params) / len(finetuned_params)
                merged_param = t * param_avg + (1 - t) * pretrained_param
                merged_params[param_name] = merged_param.reshape(out_shape)
                
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
        merge_params,
        tensor_name="default"
    ):
        out_shape = base_tensor.shape
        base_tensor = base_tensor.view(-1)
        tensors_to_merge = [
            tensor_to_merge.view(-1) 
            for tensor_to_merge in tensors_to_merge
        ]
        # follow the mergekit to implement stock merging
        delta_params = [
            tensor_to_merge - base_tensor 
            for tensor_to_merge in tensors_to_merge
        ]

        cos_thetas = []
        for i, delta_param_i in enumerate(delta_params):
            for j in range(i + 1, len(delta_params)):
                delta_param_j = delta_params[j]

                norm_product = (
                    torch.norm(delta_param_i, dim=-1) 
                    * torch.norm(delta_param_j, dim=-1)
                )
                cos_theta = (
                    (delta_param_i * delta_param_j).sum(dim=-1)
                    / norm_product.clamp(min=1e-6)
                ).clamp(min=-1, max=1)
                cos_thetas.append(cos_theta)

        # Tensor, shape (1, )
        cos_theta = torch.stack(
            cos_thetas, dim=0
        ).mean(dim=0).unsqueeze(dim=-1)
        N = len(tensors_to_merge)
        # Tensor, shape (1, )
        t = (N * cos_theta) / (1 + (N - 1) * cos_theta)

        param_avg = sum(tensors_to_merge) / len(tensors_to_merge)
        merged_param = t * param_avg + (1 - t) * base_tensor
        merged_tensor = merged_param.reshape(out_shape)
        return merged_tensor
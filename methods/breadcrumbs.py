from collections import defaultdict

import torch
from tqdm import tqdm

from utils import TaskVector
from .base_method import MergeMethod


class BreadcrumbsMerging(MergeMethod):
    def mask_smallest_largest_magnitude_param_values(
        self,
        flattened_models_to_merge_param,
        param_density=0.9,
        param_value_mask_rate=0.8
    ):
        """
        mask the smallest- and largest-magnitude parameter values (set to zeros) based on param_density and param_value_mask_rate
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
        :param param_density: float, density of retained parameters
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # num_models_to_merge, num_params = flattened_models_to_merge_param.shape
        num_mask_params = int(
            flattened_models_to_merge_param.shape[1] 
            * (1 - param_density)
        )
        num_mask_smallest_params = int(
            flattened_models_to_merge_param.shape[1] 
            * param_value_mask_rate
        )
        num_mask_largest_params = num_mask_params - num_mask_smallest_params
        assert num_mask_smallest_params >= 0 and num_mask_largest_params >= 0
        
        # Tensor, shape (num_models_to_merge, 1), find the num_mask_smallest_params-th smallest magnitude element of all the parameters in each individual model
        kth_smallest_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_smallest_params, dim=1, keepdim=True)
        # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve
        smallest_mask = flattened_models_to_merge_param.abs() >= kth_smallest_values

        # Tensor, shape (num_models_to_merge, 1), find the (flattened_models_to_merge_param.shape[1] - num_mask_largest_params)-th smallest magnitude element of all the parameters in each individual model
        kth_largest_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=flattened_models_to_merge_param.shape[1] - num_mask_largest_params, dim=1, keepdim=True)
        # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve
        largest_mask = flattened_models_to_merge_param.abs() <= kth_largest_values

        # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve finally
        mask = smallest_mask & largest_mask

        return flattened_models_to_merge_param * mask
    
    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model,
            models_to_merge,
            exclude_param_names_regex
        )
        base_model = base_model_dict['model']
        scaling_coefficient = method_params["scaling_coefficient"]
        param_value_mask_rate = method_params["param_value_mask_rate"]
        param_density = method_params["param_density"]

        assert isinstance(scaling_coefficient, float), \
            "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [
            TaskVector(
                pretrained_model=base_model_dict['model'],
                finetuned_model=model_to_merge['model'],
                exclude_param_names_regex=exclude_param_names_regex
            ) 
            for model_to_merge in merging_model_list
        ]

        # dict, dictionary of model parameters
        merged_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in tqdm(
                models_to_merge_task_vectors[0].task_vector_param_dict
            ):
                # Tensor, original shape
                param_original_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
                # Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
                flattened_models_to_merge_param = torch.vstack(
                    [
                        task_vector.task_vector_param_dict[param_name].flatten() 
                        for task_vector in models_to_merge_task_vectors
                    ]
                )
                # Tensor, shape (num_models_to_merge, num_params), mask the smallest-magnitude parameter values using param_value_mask_rate
                flattened_models_to_merge_param = self.mask_smallest_largest_magnitude_param_values(
                    flattened_models_to_merge_param=flattened_models_to_merge_param, 
                    param_density=param_density,
                    param_value_mask_rate=param_value_mask_rate
                )
                # Tensor, shape (num_params, )
                merged_flattened_param = torch.sum(
                    flattened_models_to_merge_param, 
                    dim=0
                )
                merged_task_vector_param_dict[param_name] = merged_flattened_param.reshape(param_original_shape)

            merged_task_vector = TaskVector(
                task_vector_param_dict=merged_task_vector_param_dict
            )
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(
                pretrained_model=base_model, 
                scaling_coefficient=scaling_coefficient
            )
        
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
        scaling_coefficient = method_params["scaling_coefficient"]
        param_value_mask_rate = method_params["param_value_mask_rate"]
        param_density = method_params["param_density"]
        
        assert isinstance(scaling_coefficient, float), \
            "wrong type of scaling_coefficient, should be float!"

        base_tensor_dict = {tensor_name: base_tensor}
        models_to_merge_task_vectors = [
            TaskVector(
                task_vector_param_dict={
                    tensor_name: merging_tensor - base_tensor
                }
            )
            for merging_tensor in tensors_to_merge
        ]
        
        # dict, dictionary of model parameters
        merged_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in tqdm(
                models_to_merge_task_vectors[0].task_vector_param_dict
            ):
                # Tensor, original shape
                param_original_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
                # Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
                flattened_models_to_merge_param = torch.vstack(
                    [
                        task_vector.task_vector_param_dict[param_name].flatten() 
                        for task_vector in models_to_merge_task_vectors
                    ]
                )
                # Tensor, shape (num_models_to_merge, num_params), mask the smallest-magnitude parameter values using param_value_mask_rate
                flattened_models_to_merge_param = self.mask_smallest_largest_magnitude_param_values(
                    flattened_models_to_merge_param=flattened_models_to_merge_param, 
                    param_density=param_density,
                    param_value_mask_rate=param_value_mask_rate
                )
                # Tensor, shape (num_params, )
                merged_flattened_param = torch.sum(
                    flattened_models_to_merge_param, 
                    dim=0
                )
                merged_task_vector_param_dict[param_name] = merged_flattened_param.reshape(param_original_shape)

            merged_task_vector = TaskVector(
                task_vector_param_dict=merged_task_vector_param_dict
            )
            # combine with parameters of the merged model based on scaling coefficient            
            merged_params = merged_task_vector.combine_with_base_tensor(
                base_tensor=base_tensor_dict,
                scaling_coefficient=scaling_coefficient
            )
        return merged_params[tensor_name]
import torch
import torch.nn as nn

from utils import get_param_names_to_merge, TaskVector, logger 
from .base_method import MergeMethod


class WeightedTaskVectors(MergeMethod):
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
        # method_params here are just list of scalers
        scaling_coefficients = method_params
        models_to_merge_task_vectors = [
            TaskVector(
                pretrained_model=base_model_dict['model'],
                finetuned_model=model_to_merge['model'],
                exclude_param_names_regex=exclude_param_names_regex
            )
            for model_to_merge in merging_model_list
        ]
        base_model = base_model_dict['model']
        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] * scaling_coefficients[0] \
                + models_to_merge_task_vectors[1] * scaling_coefficients[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + \
                    models_to_merge_task_vectors[index] * scaling_coefficients[index]

            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(
                pretrained_model=base_model, 
                scaling_coefficient=1.0  # Scaling coefficient is not used here, as we are using task vectors directly
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
        scaling_coefficients = method_params
        base_tensor_dict = {tensor_name: base_tensor}
        models_to_merge_task_vectors = [
            TaskVector(
                task_vector_param_dict={
                    tensor_name: merging_tensor.to("cpu") - base_tensor.to("cpu")
                }
            )
            for merging_tensor in tensors_to_merge
        ]
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] * scaling_coefficients[0] + \
                models_to_merge_task_vectors[1] * scaling_coefficients[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + \
                    models_to_merge_task_vectors[index] * scaling_coefficients[index]

            merged_params = merged_task_vector.combine_with_base_tensor(
                base_tensor=base_tensor_dict,
                scaling_coefficient=1.0
            )
        return merged_params[tensor_name]
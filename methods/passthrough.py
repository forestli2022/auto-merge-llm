import torch
import torch.nn as nn
from collections import defaultdict

from utils import get_param_names_to_merge, align_tokenizers_and_embeddings, logger
from .base_method import MergeMethod


class PassthroughMerging(MergeMethod):
    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        if base_model and not models_to_merge:
            base_model_dict, _ = self.prepare_merge(
                base_model,
                models_to_merge,
                exclude_param_names_regex
            )
            return base_model_dict['model']

        if not base_model and len(models_to_merge) == 1:
            return models_to_merge[0]['model']

        raise RuntimeError("Passthrough merge expects either a base model" 
                           "or exactly one merging model.")

    def merge_tensor(
        self, 
        base_tensor, 
        tensors_to_merge, 
        method_params, 
        mask_merging=None,
        tensor_name="default"
    ):
        if len(tensors_to_merge) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor.")

        merging_tensor = tensors_to_merge[0]
        scale = method_params.get("scale", None)
        if scale is not None:
            merging_tensor = merging_tensor * scale

        return merging_tensor

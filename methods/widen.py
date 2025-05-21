import re
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn

from utils import TaskVector, get_param_names_to_merge
from .base_method import MergeMethod


class WidenMerging(MergeMethod):
    def transpose_token_embeddings(self, param_dict):
        """
        transpose token embeddings
        """
        for param_name in param_dict:
            if param_name == "model.embed_tokens.weight":
                param_dict[param_name] = param_dict[param_name].transpose(dim0=0, dim1=1)

    def compute_param_magnitude_direction(self, param_dict, module_dict):
        """
        compute magnitude vector and direction matrix for parameters in param_dict
        :param param_dict: dict, dictionary of model parameters
        :param module_dict: dict, dictionary of model modules
        """
        param_magnitude_dict, param_direction_dict = {}, {}
        for param_name in tqdm(
            param_dict, 
            desc=f"computing parameter magnitude vector and direction matrix"
        ):
            param_last_name = param_name.split(".")[-1]
            module_name = param_name[:-len(f".{param_last_name}")]
            if param_dict[param_name].dim() == 1:
                # skip trainable vectors for norm or bias
                continue
            else:
                assert param_dict[param_name].dim() == 2
                assert (
                    isinstance(module_dict[module_name], nn.Linear) or 
                    isinstance(module_dict[module_name], nn.Embedding)
                )
                # the weight shape is (out_dim, in_dim) for both Linear and transposed nn.Embedding modules
                # Tensor, shape (in_dim, )
                magnitude_vector = torch.norm(param_dict[param_name], p=2, dim=0)
                # Tensor, shape (out_dim, in_dim)
                direction_matrix = param_dict[param_name] / magnitude_vector
                param_magnitude_dict[param_name] = magnitude_vector
                param_direction_dict[param_name] = direction_matrix

        return param_magnitude_dict, param_direction_dict

    def compute_param_magnitude_direction_differences(
        self, 
        pretrained_param_magnitude_dict, 
        pretrained_param_direction_dict,
        finetuned_param_magnitude_dict, 
        finetuned_param_direction_dict, 
        module_dict
    ):
        """
        compute the difference of magnitude and direction vectors between pretrained and finetuned models
        :param module_dict: dict, dictionary of model modules
        """
        param_magnitude_diff_dict, param_direction_diff_dict = {}, {}
        for param_name in tqdm(
            pretrained_param_magnitude_dict, 
            desc=f"computing parameter magnitude direction differences"
        ):
            matched_results_list = re.findall(
                r"model\.layers\.(\d+)\.([\w.]+)\.weight", 
                param_name
            )
            module_name = param_name[:-len(".weight")]
            if len(matched_results_list) != 0 or param_name == "lm_head.weight":
                assert isinstance(module_dict[module_name], nn.Linear)
            else:
                assert (
                    param_name == "model.embed_tokens.weight" and 
                    isinstance(module_dict[module_name], nn.Embedding)
                )

            # Tensor, shape (in_dim, )
            param_magnitude_diff = torch.abs(
                finetuned_param_magnitude_dict[param_name] 
                - pretrained_param_magnitude_dict[param_name]
            )
            # Tensor, shape (in_dim, )
            param_direction_diff = 1.0 - torch.cosine_similarity(
                finetuned_param_direction_dict[param_name], 
                pretrained_param_direction_dict[param_name], 
                dim=0
            )
            param_magnitude_diff_dict[param_name] = param_magnitude_diff
            param_direction_diff_dict[param_name] = param_direction_diff
        return param_magnitude_diff_dict, param_direction_diff_dict

    def rank_per_param_magnitude_or_direction_within_model(
        self,
        models_to_merge_param_diff
    ):
        """
        ranke the magnitude or direction within model
        :param models_to_merge_param_diff: Tensor, shape (num_models_to_merge, in_dim),
        parameter magnitude or direction difference within models that needs to be merged
        """
        # Tensor, shape (num_models_to_merge, in_dim)
        sort_indices = torch.argsort(
            models_to_merge_param_diff, 
            dim=1, 
            descending=False, 
            stable=True
        )
        # Tensor, shape (num_models_to_merge, in_dim)
        within_model_significance = (
            torch.arange(
                models_to_merge_param_diff.shape[1]
            ) / models_to_merge_param_diff.shape[1]
        ).repeat(
            models_to_merge_param_diff.shape[0]
        ).reshape(models_to_merge_param_diff.shape)
        # Tensor, shape (num_models_to_merge, in_dim)
        models_to_merge_param_within_model_significance = torch.zeros(
            within_model_significance.shape
        )

        # Tensor, shape (num_models_to_merge, in_dim)
        models_to_merge_param_within_model_significance = torch.scatter(
            input=models_to_merge_param_within_model_significance, 
            dim=1, 
            index=sort_indices, 
            src=within_model_significance
        )
        return models_to_merge_param_within_model_significance

    def compute_importance_scores(
        self, 
        input_significance_tensor, 
        above_average_value_ratio=1.0, 
        score_calibration_value=1.0
    ):
        """
        compute importance scores for input significance tensor
        :param input_significance_tensor: Tensor, shape (num_models_to_merge, in_dim), input significance tensor
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        """
        # Tensor, shape (num_models_to_merge, in_dim)
        importance_scores = torch.softmax(input_significance_tensor, dim=0)
        # assign scores for important parameters based on above average ratio
        # Tensor, shape (num_models_to_merge, 1)
        avg_input_significance_tensor = torch.mean(
            input_significance_tensor, 
            dim=1, 
            keepdim=True
        )
        # Tensor, shape (num_models_to_merge, in_dim)
        mask = (
            input_significance_tensor 
            > (
                avg_input_significance_tensor 
                * above_average_value_ratio
              )
        )
        importance_scores[mask] = score_calibration_value

        return importance_scores

    def merge_per_param_magnitude_direction(
        self,
        models_to_merge_delta_param, 
        pretrained_param,
        models_to_merge_param_magnitude_rank, 
        models_to_merge_param_direction_rank,
        above_average_value_ratio=1.0, 
        score_calibration_value=1.0
    ):
        """
        merge the magnitude and direction for each parameter
        :param models_to_merge_delta_param: Tensor, shape (num_models_to_merge, out_dim, in_dim), delta parameter of models that need to be merged
        :param pretrained_param: Tensor, shape (out_dim, in_dim), parameter of pre-trained model
        :param models_to_merge_param_magnitude_rank: Tensor, shape (num_models_to_merge, in_dim), parameter magnitude rank of models that need to be merged
        :param models_to_merge_param_direction_rank: Tensor, shape (num_models_to_merge, in_dim), parameter direction rank of models that need to be merged
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        """
        # Tensor, shape (num_models_to_merge, in_dim)
        magnitude_scores = self.compute_importance_scores(
            input_significance_tensor=models_to_merge_param_magnitude_rank,
            above_average_value_ratio=above_average_value_ratio,
            score_calibration_value=score_calibration_value
        )
        # Tensor, shape (num_models_to_merge, in_dim)
        direction_scores = self.compute_importance_scores(
            input_significance_tensor=models_to_merge_param_direction_rank,
            above_average_value_ratio=above_average_value_ratio,
            score_calibration_value=score_calibration_value
        )

        weight_scores = 0.5 * (magnitude_scores + direction_scores)
        # Tensor, shape (out_dim, in_dim)
        merged_delta_param = (
            models_to_merge_delta_param 
            * weight_scores.unsqueeze(dim=1)
        ).sum(dim=0)
        merged_param = pretrained_param + merged_delta_param

        return merged_param

    def merge_param_magnitude_direction(
        self, 
        models_to_merge_param_magnitude_direction_diff_tuples, 
        pretrained_param_dict,
        finetuned_param_dict_list,
        models_to_merge_task_vectors,
        exclude_param_names_regex,
        above_average_value_ratio=1.0,
        score_calibration_value=1.0
    ):
        """
        merge parameters by magnitudes and directions
        :param models_to_merge_param_magnitude_direction_diff_tuples: list of tuples, each tuple contains parameter magnitude and direction difference
        :param pretrained_param_dict: dict, dictionary of pretrained parameters
        :param finetuned_param_dict_list: list, list of dictionaries of finetuned parameters
        :param models_to_merge_task_vectors: list, list of task vectors
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        """
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(
            input_param_names=list(pretrained_param_dict.keys()), 
            exclude_param_names_regex=exclude_param_names_regex
        )
        # models_to_merge_param_magnitude_diff_tuple, tuple of param_magnitude_diff_dict dictionaries, length is number of models to merge
        # models_to_merge_param_direction_diff_tuple, tuple of param_direction_diff_dict dictionaries, length is number of models to merge
        (
            models_to_merge_param_magnitude_diff_tuple, 
            models_to_merge_param_direction_diff_tuple
        ) = zip(*models_to_merge_param_magnitude_direction_diff_tuples)
        param_names_merged_by_magnitude_direction = list(
            models_to_merge_param_magnitude_diff_tuple[0].keys()
        )

        merged_params = {}
        param_merging_importance_dict = {}
        for param_name in param_names_to_merge:
            # parameters that can be merged by magnitudes and directions
            if param_name in param_names_merged_by_magnitude_direction:
                # Tensor, shape (num_models_to_merge, out_dim, in_dim)
                models_to_merge_delta_param = torch.stack(
                    [
                        models_to_merge_task_vector.task_vector_param_dict[param_name]
                        for models_to_merge_task_vector in models_to_merge_task_vectors
                    ], dim=0
                )
                # Tensor, shape (num_models_to_merge, in_dim)
                models_to_merge_param_magnitude_diff = torch.stack(
                    [
                        model_to_merge_param_magnitude_diff[param_name]
                        for model_to_merge_param_magnitude_diff in models_to_merge_param_magnitude_diff_tuple
                    ], dim=0
                )
                # Tensor, shape (num_models_to_merge, in_dim)
                models_to_merge_param_direction_diff = torch.stack(
                    [
                        model_to_merge_param_direction_diff[param_name]
                        for model_to_merge_param_direction_diff in models_to_merge_param_direction_diff_tuple
                    ], dim=0
                )

                # rank magnitudes and directions within each model
                # Tensor, shape (num_models_to_merge, in_dim)
                models_to_merge_param_magnitude_rank = (
                    self.rank_per_param_magnitude_or_direction_within_model(
                        models_to_merge_param_diff=models_to_merge_param_magnitude_diff
                    )
                )
                models_to_merge_param_direction_rank = (
                    self.rank_per_param_magnitude_or_direction_within_model(
                        models_to_merge_param_diff=models_to_merge_param_direction_diff
                    )
                )

                # Tensor, shape (out_dim, in_dim)
                merged_params[param_name] = (
                    self.merge_per_param_magnitude_direction(
                        models_to_merge_delta_param=models_to_merge_delta_param,
                        pretrained_param=pretrained_param_dict[param_name],
                        models_to_merge_param_magnitude_rank=models_to_merge_param_magnitude_rank,
                        models_to_merge_param_direction_rank=models_to_merge_param_direction_rank,
                        above_average_value_ratio=above_average_value_ratio,
                        score_calibration_value=score_calibration_value
                    )
                )
            # parameters that not required to be merged by magnitudes and directions (vector-like weights like the normalization layers)
            else:
                # Tensor, shape (num_models_to_merge, in_dim)
                models_to_merge_param = torch.stack(
                    [
                        finetuned_param_dict[param_name] 
                        for finetuned_param_dict in finetuned_param_dict_list
                    ], dim=0
                )
                # Tensor, shape (num_models_to_merge, in_dim)
                models_to_merge_delta_param = torch.stack(
                    [
                        models_to_merge_task_vector.task_vector_param_dict[param_name]
                        for models_to_merge_task_vector in models_to_merge_task_vectors
                    ], dim=0
                )
                param_diff = torch.abs(
                    models_to_merge_param 
                    - pretrained_param_dict[param_name]
                )

                # Tensor, shape (num_models_to_merge, in_dim)
                param_scores = self.compute_importance_scores(
                    input_significance_tensor=param_diff, 
                    above_average_value_ratio=above_average_value_ratio,
                    score_calibration_value=score_calibration_value
                )

                # Tensor, shape (in_dim, )
                merged_delta_param = (
                    models_to_merge_delta_param * param_scores
                ).sum(dim=0)
                merged_params[param_name] = (
                    pretrained_param_dict[param_name] 
                    + merged_delta_param
                )

        return merged_params
    
    def merge(
        self, 
        base_model, 
        models_to_merge, 
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        """
        widen merging method based on weight disentanglement
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        :return:
        """
        
        above_average_value_ratio = method_params["above_average_value_ratio"]
        score_calibration_value = method_params["score_calibration_value"]
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, 
            models_to_merge, 
            exclude_param_names_regex
        )
        base_model = base_model_dict['model']
        module_dict = {
            module_name: module 
            for module_name, module in base_model.named_modules()
        }
        pretrained_param_dict = {
            param_name: param_value 
            for param_name, param_value in base_model.named_parameters()
        }
        finetuned_param_dict_list = [
            {
                param_name: param_value 
                for param_name, param_value in model_to_merge['model'].named_parameters()
            } for model_to_merge in merging_model_list
        ]
        models_to_merge_task_vectors = [
            TaskVector(
                pretrained_model=base_model, 
                finetuned_model=model_to_merge['model'], 
                exclude_param_names_regex=exclude_param_names_regex
            ) for model_to_merge in merging_model_list
        ]

        # transpose token embeddings
        self.transpose_token_embeddings(param_dict=pretrained_param_dict)
        for finetuned_param_dict in finetuned_param_dict_list:
            self.transpose_token_embeddings(param_dict=finetuned_param_dict)
        for models_to_merge_task_vector in models_to_merge_task_vectors:
            self.transpose_token_embeddings(
                param_dict=models_to_merge_task_vector.task_vector_param_dict
            )

        with torch.no_grad():
            models_to_merge_param_magnitude_direction_diff_tuples = []
            # compute the magnitude vectors, direction matrices and the difference compared to pre-trained model
            (
                pretrained_param_magnitude_dict, 
                pretrained_param_direction_dict 
            ) = self.compute_param_magnitude_direction(
                param_dict=pretrained_param_dict, 
                module_dict=module_dict
            )
            for finetuned_param_dict in finetuned_param_dict_list:
                (
                    finetuned_param_magnitude_dict, 
                    finetuned_param_direction_dict 
                ) = self.compute_param_magnitude_direction(
                    param_dict=finetuned_param_dict, 
                    module_dict=module_dict
                )
                param_magnitude_direction_diff_tuple = (
                    self.compute_param_magnitude_direction_differences(
                        pretrained_param_magnitude_dict=pretrained_param_magnitude_dict,
                        pretrained_param_direction_dict=pretrained_param_direction_dict,
                        finetuned_param_magnitude_dict=finetuned_param_magnitude_dict,
                        finetuned_param_direction_dict=finetuned_param_direction_dict,
                        module_dict=module_dict
                    )
                )
                models_to_merge_param_magnitude_direction_diff_tuples.append(
                    param_magnitude_direction_diff_tuple
                )

            # merge parameters based on computed difference
            merged_params = self.merge_param_magnitude_direction(
                models_to_merge_param_magnitude_direction_diff_tuples=models_to_merge_param_magnitude_direction_diff_tuples,
                pretrained_param_dict=pretrained_param_dict,
                finetuned_param_dict_list=finetuned_param_dict_list,
                models_to_merge_task_vectors=models_to_merge_task_vectors,
                exclude_param_names_regex=exclude_param_names_regex,
                above_average_value_ratio=above_average_value_ratio,
                score_calibration_value=score_calibration_value
            )
            # transpose token embeddings to recover
            self.transpose_token_embeddings(param_dict=merged_params)

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
        raise NotImplementedError("This function is not yet supported.")
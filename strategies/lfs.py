import gc
import json
import time
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from transformers import AutoConfig, AutoTokenizer

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from smac import (
    MultiFidelityFacade,
    Scenario,
    RunHistory,
    HyperparameterOptimizationFacade,
)
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.runhistory import TrialInfo, TrialValue
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity

from evaluation import evaluator_classes
from utils import logger
from .base_strategy import MergeStrategy
from .merge_utils import MergeUtils

SUPPORTED_METHOD_PARAM_MAPS = {
    "linear": ["weights"],
    "task_arithmetic": ["scaling_coefficient"],
    "ties": ["param_value_mask_rate", "scaling_coefficient"],
    "slerp": ["slerp_t" ]
}


class LfsMerge(MergeStrategy):
    def __init__(self, config):
        super().__init__(config) 
        self.models = self.config["models"]
        self.merging_method = self.config["merging_method"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.random_init_points = self.config.get("random_init_points", 0)
        self.base_model = self.config["base_model"]
        self.layer_granularity = self.config["layer_granularity"]
        self.n_trials = self.config["n_trials"]
        self.min_budget = self.config.get("min_budget", 20)
        self.max_budget = self.config.get("max_budget", 1000)
        self.total_budget = self.config.get("total_budget")
        self.eta = self.config.get("eta", 3)
        self.n_workers = self.config.get("n_workers", 4)
        self.filters = self.config.get("filters",[])
        self.n_param_sets = self.filters + ['base']
        self.output_path = self.config.get("output_path", None)
        
        # Evaluation setup
        self.evaluate_tasks = [task['task'] for task in 
                               self.config.get('evaluation', {}).get('tasks', [])]
        self.in_memory_evaluate = self.config.get('evaluation', {}).get('in_memory', False)
        evaluator_key = 'inmemory_evaluate' if self.in_memory_evaluate else 'ondisk_evaluate'
        self.evaluator_class = evaluator_classes[evaluator_key]
        self.evaluator_instance = self.evaluator_class(self.config)
        
        # Model architecture info
        self.num_layers = self.get_config().num_hidden_layers 
        self.num_groups = self.num_layers // self.layer_granularity
        if self.layer_granularity > 0:
            assert self.num_groups * self.layer_granularity == self.num_layers
    
    def get_config(self):
        if self.base_model:
            sample_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.base_model
            )
            return sample_config
        sample_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.models[0]
            )
        return sample_config
    
    def get_filter_t_values(self, group_index, config):
        """Extract t-values for slerp from configuration."""
        filter_t_values = {}
        for cur_model in self.models:
            for filter_type in self.n_param_sets: 
                filter_t_key = f"model_{cur_model}_layer_{group_index}_paramset_{filter_type}_method_slerp_param_slerp_t"
                filter_t_values[(cur_model, filter_type)] = config.get(filter_t_key, 0)
        return filter_t_values
    
    def calculate_filter_weights(self, t_values, model1, model2):
        """Calculate weights for each filter based on t-values."""
        filter_weights = {}
        for filter_type in self.n_param_sets:
            t1 = t_values.get((model1, filter_type), 0)
            t2 = t_values.get((model2, filter_type), 0)
            weight1, weight2 = np.exp([t1, t2]) / np.sum(np.exp([t1, t2]))
            filter_weights[filter_type] = {"value": weight2}
        return filter_weights

    def get_slerp_slices(self, group_index, layer_range, config):
        """Generate merging configuration for slerp method."""
        sources = []
        t_values = self.get_filter_t_values(group_index, config)
        
        model_scores = {}
        for (model, filter_type), t in t_values.items():
            if model not in model_scores:
                model_scores[model] = 0
            model_scores[model] += t 

        # pick top2 model
        sorted_models = sorted(model_scores, key=model_scores.get, reverse=True)
        model1, model2 = sorted_models[:2]

        sources.append({
            "model": model1,
            "layer_range": layer_range,
        })
        sources.append({
            "model": model2,
            "layer_range": layer_range,
        })

        merging_params = {
            "slerp_t": []
        }
        filter_weights = self.calculate_filter_weights(t_values, model1, model2)

        for filter_type, weights in filter_weights.items():
            if filter_type=="base":
                merging_params["slerp_t"].append({"value": weights['value']})
            else:   
                merging_params["slerp_t"].append({"filter": filter_type, "value": weights['value']})
            
        slice = {
            "sources": sources,
            "merging_method": {
                "slerp": merging_params
            }
        }
        return slice

    def get_inital_params(self):
        init_trials = []
        methods_list = SUPPORTED_METHOD_PARAM_MAPS.keys()
        def get_single_model_config(model_index):
            config = {}
            for group_index in range(self.num_groups):
                config[f"layer_selection_method_{group_index}"] = "linear"
            
            for group_index in range(self.num_groups):
                for cur_param_set in self.n_param_sets:
                    for method in methods_list:
                        method_config = self.merging_method[method]
                        for method_param in SUPPORTED_METHOD_PARAM_MAPS[method]:
                            min_value = method_config[method_param]['min']
                            max_value = method_config[method_param]['max']
                            if method == "linear" or method == "slerp":
                                for idx, model in enumerate(self.models):
                                    if idx == model_index and method == "linear":
                                        config[f"model_{model}_layer_{group_index}_paramset_{cur_param_set}_method_{method}_param_{method_param}"] = 1.0
                                    else:
                                        config[f"model_{model}_layer_{group_index}_paramset_{cur_param_set}_method_{method}_param_{method_param}"] = 0.0
                            else:
                                if method_param == "param_value_mask_rate":
                                    config[f"layer_{group_index}_paramset_{cur_param_set}_method_{method}_param_{method_param}"] = 0.1
                                else:  
                                    config[f"layer_{group_index}_paramset_{cur_param_set}_method_{method}_param_{method_param}"] = 0.0                   
            return config
        init_trials = [get_single_model_config(model_index) for model_index in range(len(self.models))]      
        return init_trials
             
    def generate_genotype(self, config):
        """Generate layer slices configuration from optimization parameters."""
        slices = []
        layer_step = (
            self.layer_granularity
            if self.layer_granularity > 0
            else self.num_layers
        )             
        for group_index in range(self.num_groups):
            layer_range = [group_index * layer_step, (group_index + 1) * layer_step]
            selected_method = config.get(f"layer_selection_method_{group_index}", "linear")
            # Handle slerp method
            if selected_method == "slerp":
                slice = self.get_slerp_slices(group_index, layer_range, config)
            else:
                # Handle other methods
                avg_parameters = {}
                sources = []
                # Collect parameters for each method type
                for param_name in SUPPORTED_METHOD_PARAM_MAPS[selected_method]:
                    avg_parameters[param_name] = {}
                    for filter_type in self.n_param_sets:
                        param_values = []
                        if selected_method == "linear":
                            # For linear, collect weights for each model
                            for cur_model in self.models:
                                param_key = f"model_{cur_model}_layer_{group_index}_paramset_{filter_type}_method_{selected_method}_param_{param_name}"
                                param_value = config.get(param_key, 0)
                                param_values.append(param_value)
                            res_value = param_values
                        else:
                            # For other methods, get the single parameter value
                            param_key = f"layer_{group_index}_paramset_{filter_type}_method_{selected_method}_param_{param_name}"
                            param_value = config.get(param_key, 0)
                            param_values.append(param_value)
                            res_value = param_values[0]
                        # Format parameters based on filter type
                        if filter_type == "base":
                            avg_parameters[param_name][filter_type] = {"value": res_value}
                        else:
                            avg_parameters[param_name][filter_type] = {"filter": filter_type, "value": res_value}
                
                slice = {
                    "sources": [{"model": cur_model, "layer_range": layer_range} for cur_model in self.models],
                    "merging_method": {
                        selected_method: 
                            {
                                param_name: list(filters.values()) 
                                for param_name, filters in avg_parameters.items()
                            }  
                    }
                }
            slices.append(slice)
        return slices 

    def get_config_space(self):
        """Define the configuration space for hyperparameter optimization."""
        cs = ConfigurationSpace()
        methods_list = SUPPORTED_METHOD_PARAM_MAPS.keys()
        methods_params_list = [
            f"{method}_{param}"
            for method, params in SUPPORTED_METHOD_PARAM_MAPS.items()
            for param in params
        ]
        layer_step = (
            self.layer_granularity
            if self.layer_granularity > 0
            else self.num_layers
        )
        
        config_list = []
        
        # Add selection methods for each layer group
        for group_index in range(self.num_groups):
            config_list.append(Categorical(
                f"layer_selection_method_{group_index}", 
                items=methods_list
            ))
        
        # Add parameters for each method and filter type
        for group_index in range(self.num_groups):
            for cur_param_set in self.n_param_sets:
                for method in methods_list:
                    method_config = self.merging_method[method]
                    
                    for method_param in SUPPORTED_METHOD_PARAM_MAPS[method]:
                        min_value = method_config[method_param]['min']
                        max_value = method_config[method_param]['max']
                        if method == "linear" or method == "slerp":
                            # Model-specific parameters for linear and slerp
                            for model in self.models:
                                param_name = f"model_{model}_layer_{group_index}_paramset_{cur_param_set}_method_{method}_param_{method_param}"
                                config_list.append(Float(param_name, (min_value, max_value)))
                        else:
                            # Layer-specific parameters for other methods
                            param_name = f"layer_{group_index}_paramset_{cur_param_set}_method_{method}_param_{method_param}"
                            config_list.append(Float(param_name, (min_value, max_value)))
                            
        cs.add_hyperparameters(config_list)
        return cs
                    
    def objective(self, config, seed, budget):
        """Objective function for hyperparameter optimization."""
        logger.info(f"start evaluating, current budget is {budget}")
        budget = int(budget)
        result = {}
        slices = self.generate_genotype(config)
        logger.info(slices)
        
        merge_utils = MergeUtils(
            base_model=self.base_model,
            merging_models=None, 
            merging_method=None, 
            slices=slices, 
            model_storage_path=self.output_path,
            in_memory=self.in_memory_evaluate, 
        )
        merge_utils.merge_slices()
        
        try:
            if self.in_memory_evaluate:
                out_tensors = merge_utils.out_tensors
                output_config = merge_utils.output_config
                aligned_tokenizer = merge_utils.aligned_tokenizer
                logger.info(f"current layer is {output_config.num_hidden_layers}")
                eval_result = self.evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer, budget)     
            else:
                eval_result = self.evaluator_instance.evaluate(self.output_path, budget)  
            # Calculate error rate (1 - score)
            for cur_task in self.evaluate_tasks:
                result[cur_task]=1-eval_result[cur_task]['score']  
            # Clean up resources
            del out_tensors
            del merge_utils._out_tensors
            del merge_utils
            try:
                self.evaluator_instance._destroy_llm()
                self.evaluator_instance._clean_inner_model()
            except:
                logger.error("fail to destroy llm")
                logger.error(traceback.format_exc())
            gc.collect()  
        except Exception as e:
            logger.info(traceback.format_exc())
            try:
                # Clean up resources
                if self.in_memory_evaluate:
                    del out_tensors
                    del merge_utils._out_tensors
                
                self.evaluator_instance._destroy_llm()
                del merge_utils
                gc.collect()  
            except:
                logger.error("fail to eval and clean fail")
                logger.error(traceback.format_exc())
            # Return worst score on error
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1
        return result[self.evaluate_tasks[0]]

    
    def optimize(self):
        """Run hyperparameter optimization to find optimal merging parameters."""
        configspace = self.get_config_space()
        # Set up GPU cluster for parallel evaluation
        logger.info(",".join(map(str, range(min(self.n_workers, torch.cuda.device_count())))))
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(min(self.n_workers, torch.cuda.device_count())))),
            threads_per_worker=1,
            memory_limit="90GB",
            device_memory_limit=0.9
        )
        client = Client(cluster)
        logger.info(f"Client: {client}")
    
        if self.n_trials == 0:
            self.n_trials = get_n_trials_for_hyperband_multifidelity(
                total_budget=self.total_budget,  
                min_budget=self.min_budget, 
                max_budget=self.max_budget, 
                eta=self.eta, 
                print_summary=True,
            )
        
        # Scenario object specifying the optimization "environment"
        scenario = Scenario(
            configspace,  
            output_directory=Path(self.output_path), 
            deterministic=True, 
            n_trials=self.n_trials, 
            min_budget=self.min_budget, 
            max_budget=self.max_budget
        )
        
        intensifier = MultiFidelityFacade.get_intensifier(scenario=scenario, eta=self.eta)
        if self.load_run_history != None:
            runhistory = RunHistory()
            runhistory.update_from_json(self.load_run_history, configspace)
            initial_design=MultiFidelityFacade.get_initial_design(
                scenario, 
                n_configs=0,
                additional_configs=None,
            )
            smac = MultiFidelityFacade(
                scenario,
                self.objective,
                overwrite=False,
                intensifier=intensifier,
                logging_level=0,
                initial_design=initial_design,
                dask_client=client
            )
            for (trial_key, trial_value) in runhistory.items():
                trial_info = TrialInfo(
                    config=runhistory.get_config(trial_key.config_id),
                    instance=trial_key.instance,
                    seed=trial_key.seed,
                    budget=trial_key.budget
                )
                smac.tell(trial_info, trial_value) 
        else:
            init_trials = self.get_inital_params()
            configurations = [Configuration(configspace, trial) for trial in init_trials]
            initial_design=MultiFidelityFacade.get_initial_design(
                scenario, 
                n_configs=self.random_init_points,
                additional_configs=configurations,
            )
      
            smac = MultiFidelityFacade(
                scenario,
                self.objective,
                overwrite=False,
                intensifier=intensifier,
                initial_design=initial_design,
                logging_level=0,
                dask_client=client
            )

        incumbent = smac.optimize()
        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost: {incumbent_cost}")

    def merge(self):
        self.optimize()
    
    def eval_config(self, config, config_id=0):
        result = {}
        configspace = self.get_config_space()
        config = Configuration(configspace, config)
        slices = self.generate_genotype(config)
        logger.info(f"Evaulating current slices :{slices}")
        merge_utils = MergeUtils(base_model=self.base_model,
                                 merging_models=None, 
                                 merging_method=None, 
                                 slices=slices, 
                                 model_storage_path=self.output_path,
                                 in_memory=self.in_memory_evaluate, 
                                )
        merge_utils.merge_slices()
        
        try:
            if self.in_memory_evaluate:
                out_tensors = merge_utils.out_tensors
                output_config = merge_utils.output_config
                aligned_tokenizer = merge_utils.aligned_tokenizer
                logger.info(f"current layer is {output_config.num_hidden_layers}")
                result = self.evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer)      
            else:
                result = self.evaluator_instance.evaluate(self.output_path)  
            
            del out_tensors
            del merge_utils._out_tensors
            del merge_utils
            gc.collect()    
        except Exception as e:
            logger.info(traceback.format_exc())
            result['score']=0   
        return result
        
if __name__ == "__main__":
    pass

import gc
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from transformers import AutoConfig, AutoTokenizer

from ConfigSpace import (
    AndConjunction,
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
    NotEqualsCondition,
)
from ConfigSpace.conditions import EqualsCondition, GreaterThanCondition, InCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac import (
    HyperparameterOptimizationFacade,
    RunHistory,
    Scenario,
)
from smac.intensifier.hyperband import Hyperband
from smac.multi_objective.parego import ParEGO
from smac.runhistory import TrialInfo, TrialValue

# Local application imports
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


class FoldMerge(MergeStrategy):
    def __init__(self, config):
        super().__init__(config) 
        logger.info(f"config : {self.config}")
        
        # Extract configuration parameters
        self.models = self.config["models"]
        self.base_model = self.config["base_model"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.random_init_points = self.config.get("random_init_points", 0)
        self.num_hidden_layers = self.config["num_hidden_layers"]
        self.candidate_layers = int(self.num_hidden_layers)
        self.candidates_per_layer = len(self.models)
        self.max_layers = self.config.get("layers", 40)
        self.remove_layers = self.num_hidden_layers - self.max_layers
        self.evaluate_tasks = [task['task'] for task in self.config.get('evaluation', {}).get('tasks', [])]
        
        # Optimization parameters
        self.n_workers = config.get("n_workers", 1)
        self.n_trials = config.get("n_trials")
        self.min_budget = config.get("min_budget", 20)
        self.max_budget = config.get("max_budget", 1000)
        self.total_budget = config.get("total_budget")
        self.eta = config.get("eta", 3)
        
        # Merging configuration    
        self.merging_method = config.get("merging_method", "passthrough")
        self.output_path = self.config.get("output_path", None)
        
        # Evaluation setup
        self.in_memory_evaluate = (
            self.config.get('evaluation', {}).get('in_memory', False)
        )
        self.evaluator_class = (
            evaluator_classes['inmemory_evaluate']
            if self.in_memory_evaluate 
            else evaluator_classes['ondisk_evaluate']
        )
        self.evaluator_instance = self.evaluator_class(self.config)
        

    def generate_genotype(self, config):
        """Generate model configuration from hyperparameters."""
        if self.remove_layers != 0:
            remove_indices = [config[f'remove_idx_{i}'] for i in range(self.remove_layers)] 
        else:
            remove_indices = []
        
        slices = []
        output_scales = []
        # From layer 0 to layer num_hidden_layers-1
        for layer_idx in range(self.candidate_layers):
            
            target = {
                "model": self.base_model,
                "layer_range": [layer_idx, layer_idx + 1]
            }
            
            layer_scale = config.get(f'layer_{layer_idx}_output_scale', 1)
            output_scales.append(layer_scale)
            
            candidate_layer = []
            for cand_idx, model in enumerate(self.models):
                if f'layer_{layer_idx}_candidate_{cand_idx}' in config.keys():
                    candidate_layer.append(model)
            
            merge_scale = config.get(f'layer_{layer_idx}_merge_scale_factor', 1)
            if layer_idx in remove_indices:
                collapse_scale = config.get(f'layer_{layer_idx}_collapse_scale_factor', 1.0)
                merge_collapse_order = config.get(f'layer_{layer_idx}_merge_collapse_order', 0)
            
            if len(candidate_layer) == 0:
                slice_dict = {
                        "sources": [
                            {
                                "model": self.base_model,
                                "layer_range": [
                                    layer_idx,
                                    layer_idx + 1,
                                ],
                            }
                        ],
                        "merging_method": {"passthrough": {"scale": [{"value": 1.0}]}},
                    }
            
            elif len(candidate_layer) == 1:
                slice_dict = {
                        "sources": [
                            {
                                "model": candidate_layer[0],
                                "layer_range": [
                                    layer_idx,
                                    layer_idx + 1,
                                ],
                            }
                        ],
                        "merging_method": {"passthrough": {"scale": [{"value": 1.0}]}},
                    }
            
            else:
                sources = []
                for current_model in candidate_layer:
                    sources.append(
                        {
                            "model": current_model,
                            "layer_range": [
                                layer_idx,
                                layer_idx + 1,
                            ],
                        }
                    )
                slice_dict = {
                    "sources": sources,
                    "target": target,
                    "merging_method": {"task_arithmetic": {"scaling_coefficient": [{"value": merge_scale}]}},
                }
                    

            # Add collapse scale factor and merge/collapse order if the layer is in the remove list
            if layer_idx in remove_indices:
                slice_dict["collapsing_method"] = {
                    "task_arithmetic": {
                        "scaling_coefficient": [{"value": collapse_scale}]
                    }
                }
                slice_dict["merge_collapse_order"] = merge_collapse_order
            slices.append(slice_dict)
        
        return slices, None  

                 
    def objective(self, config, seed, budget):
        """Objective function for hyperparameter optimization."""
        # You could also adjust budget here
        
        result = {}
        # Generate model configuration
        slices, output_scales = self.generate_genotype(config)
        logger.info(f"current genotype : {slices}")
        
        # Create merged model
        merge_utils = MergeUtils(
            base_model=self.base_model,
            merging_models=None, 
            merging_method=None, 
            slices=slices, 
            model_storage_path=self.output_path,
            in_memory=self.in_memory_evaluate,
            output_scales=output_scales
        )
        merge_utils.fold_slices()
        try:
            if self.in_memory_evaluate:
                out_tensors = merge_utils.out_tensors
                output_config = merge_utils.output_config
                aligned_tokenizer = merge_utils.aligned_tokenizer
                eval_result = self.evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer, budget)  
            else:
                eval_result = self.evaluator_instance.evaluate(self.output_path, budget)  
            # Manually release memory here to address SMAC's behavior of creating a new instance each time.
            # Todo: This causes the initialization of vllm each time, which can be time-consuming
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1 - eval_result[cur_task]['score']
            
            # Clean up resources
            if self.in_memory_evaluate:
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
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1
            try:
                if self.in_memory_evaluate:
                    del out_tensors
                del merge_utils._out_tensors
                del merge_utils
                self.evaluator_instance._destroy_llm()
                self.evaluator_instance._clean_inner_model()
                gc.collect()  
            except:
                logger.error("fail to eval and clean fail")
                logger.error(traceback.format_exc())
        return result


    def get_config(self):
        """Define the configuration space for optimization."""
        cs = ConfigurationSpace()
        remove_count = self.remove_layers
        
        # Generate layer removal strategy
        remove_list = self.generate_remove_lists(self.num_hidden_layers, remove_count)["alternating"]

        # Add hyperparameters for layers to remove
        if remove_count != 0:
            for i in range(remove_count):
                cs.add_hyperparameter(UniformIntegerHyperparameter(
                    f'remove_idx_{i}', 
                    lower=0, 
                    upper=self.num_hidden_layers-1, 
                    default_value=remove_list[i],
                ))

            # Add constraints to prevent duplicate removals
            for i in range(remove_count-1):
                for j in range(i+1, remove_count):
                    for val in range(self.num_hidden_layers):
                        cs.add_forbidden_clause(
                            ForbiddenAndConjunction(
                                ForbiddenEqualsClause(cs.get_hyperparameter(f'remove_idx_{i}'), val),
                                ForbiddenEqualsClause(cs.get_hyperparameter(f'remove_idx_{j}'), val)
                            )
                        )

        # Add hyperparameters for each layer
        for layer_idx in range(self.num_hidden_layers):
            # Add candidate selection parameters
            for cand_idx in range(self.candidates_per_layer):
                candidate_param = CategoricalHyperparameter(
                    f'layer_{layer_idx}_candidate_{cand_idx}', 
                    choices=[0, 1],
                    default_value=1 if cand_idx == 0 else 0  
                )
                cs.add_hyperparameter(candidate_param)

                # Add conditions for layer removal
                if remove_count!=0:
                    candidate_conditions = []
                    for i in range(remove_count):
                        remove_idx_param = cs.get_hyperparameter(f'remove_idx_{i}')
                        condition = NotEqualsCondition(
                            candidate_param,  
                            remove_idx_param,   
                            layer_idx           
                        )
                        candidate_conditions.append(condition)
                
                    if candidate_conditions:
                        candidate_condition = AndConjunction(*candidate_conditions)
                        cs.add_condition(candidate_condition)
            
            # Add merging method parameters
            method_config = self.merging_method["task_arithmetic"]
            for method_param in SUPPORTED_METHOD_PARAM_MAPS["task_arithmetic"]: 
                min_value = method_config[method_param]['min']
                max_value = method_config[method_param]['max']
            
            # Add scale factor parameter
            scale_factor_param = UniformFloatHyperparameter(
                f'layer_{layer_idx}_merge_scale_factor', 
                lower=min_value, 
                upper=max_value, 
                default_value=1.0
            )
            cs.add_hyperparameter(scale_factor_param)
            
            # Add conditions
            # if remove_count!=0:
            #     merge_method_conditions = []
            #     for i in range(remove_count):
            #         remove_idx_param = cs.get_hyperparameter(f'remove_idx_{i}')
            #         condition = NotEqualsCondition(
            #             scale_factor_param,  
            #             remove_idx_param,   
            #             layer_idx           
            #         )
            #         merge_method_conditions.append(condition)
            #     if merge_method_conditions:
            #             merge_method_condition = AndConjunction(*merge_method_conditions)
            #             cs.add_condition(merge_method_condition)    
            
            # Add collapse scale factor parameter
            collapse_scale_param = UniformFloatHyperparameter(
                f'layer_{layer_idx}_collapse_scale_factor', 
                lower=min_value, 
                upper=max_value, 
                default_value=1.0
            )
            cs.add_hyperparameter(collapse_scale_param)

            # Add conditions
            if remove_count!=0:
                collapse_conditions = []
                for i in range(remove_count):
                    remove_idx_param = cs.get_hyperparameter(f'remove_idx_{i}')
                    condition = EqualsCondition(
                        collapse_scale_param,  
                        remove_idx_param,   
                        layer_idx            
                    )
                    collapse_conditions.append(condition)
                
                if collapse_conditions:
                    collapse_condition = AndConjunction(*collapse_conditions)
                    cs.add_condition(collapse_condition)
            
            # Add merge collapse order parameter
            merge_collapse_order_param = CategoricalHyperparameter(
                f'layer_{layer_idx}_merge_collapse_order', 
                choices=[0, 1],  # 0 for merge (i-1)th layer first, 1 for collapse first
                default_value=0
            )
            cs.add_hyperparameter(merge_collapse_order_param)

            # Add conditions
            if remove_count!=0:
                merge_collapse_order_conditions = []
                for i in range(remove_count):
                    remove_idx_param = cs.get_hyperparameter(f'remove_idx_{i}')
                    condition = EqualsCondition(
                        merge_collapse_order_param,  
                        remove_idx_param,   
                        layer_idx            
                    )
                    merge_collapse_order_conditions.append(condition)
                
                if merge_collapse_order_conditions:
                    merge_collapse_order_condition = AndConjunction(*merge_collapse_order_conditions)
                    cs.add_condition(merge_collapse_order_condition)

            # Add output scale parameter
            output_scale_param = UniformFloatHyperparameter(
                f'layer_{layer_idx}_output_scale', 
                lower=0, 
                upper=2.0, 
                default_value=1.0
            )
            cs.add_hyperparameter(output_scale_param)
            
            # Add conditions
            if remove_count!=0:
                output_scale_conditions = []
                for i in range(remove_count):
                    remove_idx_param = cs.get_hyperparameter(f'remove_idx_{i}')
                    condition = NotEqualsCondition(
                        output_scale_param,  
                        remove_idx_param,   
                        layer_idx            
                    )
                    output_scale_conditions.append(condition)
                
                if output_scale_conditions:
                    output_scale_condition = AndConjunction(*output_scale_conditions)
                    cs.add_condition(output_scale_condition)

        return cs 


    def generate_remove_lists(self, num_hidden_layers, remove_count, seed=42):
        """Generate strategies for removing layers."""
        random.seed(seed)
        
        options = {}
        
        # Calculate key points
        mid = num_hidden_layers // 2
        
        # Create basic ranges
        higher = list(range(mid, num_hidden_layers - 1))  # Higher layers (excluding the highest)
        lower = list(range(2, mid))                       # Lower layers (excluding the lowest)
        extremes = [num_hidden_layers - 1, 1]             # Highest and lowest layers
        
        # Option 1: Higher layers first, then lower
        higher_first = higher.copy()
        random.shuffle(higher_first)
        lower_copy = lower.copy()
        random.shuffle(lower_copy)
        
        remove_list = higher_first + lower_copy + extremes
        options["higher_first"] = remove_list[:remove_count]
        
        # Option 2: Middle outward
        middle_layers = list(range(mid - 3, mid + 4))  # Layers around the middle
        random.shuffle(middle_layers)
        
        remaining = [i for i in range(1, num_hidden_layers - 1) if i not in middle_layers]
        random.shuffle(remaining)
        
        remove_list = middle_layers + remaining + extremes
        options["middle_outward"] = remove_list[:remove_count]
        
        # Option 3: Alternating high-low
        alternating = []
        high_temp = higher.copy()
        low_temp = lower.copy()
        random.shuffle(high_temp)
        random.shuffle(low_temp)
        
        # Interleave high and low layers
        max_length = max(len(high_temp), len(low_temp))
        for i in range(max_length):
            if i < len(high_temp):
                alternating.append(high_temp[i])
            if i < len(low_temp):
                alternating.append(low_temp[i])
        
        alternating.extend(extremes)
        options["alternating"] = alternating[:remove_count]
        
        return options

    def get_initial_params(self):
        """Generate initial parameters for optimization."""
        initial_params = []
        total_layers = self.num_hidden_layers
        remove_count = self.remove_layers
        
        # Generate remove_list options with simplified logic
        remove_list_options = self.generate_remove_lists(total_layers, remove_count)
        
        # Merge factors and candidate patterns
        merge_scales = [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
        
        candidate_patterns = [
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1], 
            [1, 1, 1],
        ]

        # Generate initial parameters for each remove_list option
        for option_name, remove_list in remove_list_options.items():
            # Default configuration
            config_dict = {}
            
            if len(remove_list) != 0:
                for i, idx in enumerate(remove_list):
                    config_dict[f'remove_idx_{i}'] = idx
                    
            for layer_idx in range(total_layers):
                config_dict[f'layer_{layer_idx}_output_scale'] = 1.0
                config_dict[f'layer_{layer_idx}_merge_scale_factor'] = 1.0
                
                # For each layer in the remove list, set collapse scale factor (and merge method, default to TA)
                if layer_idx in remove_list:
                    config_dict[f'layer_{layer_idx}_collapse_scale_factor'] = 1.0
                    # Randomly set merge/collapse order: 0 for merge (i-1)th layer first, 1 for collapse first
                    config_dict[f'layer_{layer_idx}_merge_collapse_order'] = random.randint(0, 1)

                for cand_idx in range(self.candidates_per_layer):
                    config_dict[f'layer_{layer_idx}_candidate_{cand_idx}'] = 0
                        
            initial_params.append(config_dict)
            
            # Generate configurations for different merge factors and candidate patterns
            for merge_scale in merge_scales:
                for candidate_pattern in candidate_patterns:
                    config_dict = {}
                    
                    if len(remove_list) != 0:
                        for i, idx in enumerate(remove_list):
                            config_dict[f'remove_idx_{i}'] = idx
                            
                    for layer_idx in range(total_layers):
                        config_dict[f'layer_{layer_idx}_output_scale'] = 1.0
                        config_dict[f'layer_{layer_idx}_merge_scale_factor'] = merge_scale

                        # For each layer in the remove list, set collapse scale factor (and merge method, default to TA)
                        if layer_idx in remove_list:
                            # FIXME: Find a way to set collapse factor
                            config_dict[f'layer_{layer_idx}_collapse_scale_factor'] = 0.3
                            # Randomly set merge/collapse order: 0 for merge (i-1)th layer first, 1 for collapse first
                            config_dict[f'layer_{layer_idx}_merge_collapse_order'] = random.randint(0, 1)
                        
                        for cand_idx in range(self.candidates_per_layer):
                            config_dict[f'layer_{layer_idx}_candidate_{cand_idx}'] = candidate_pattern[cand_idx]
                                
                    initial_params.append(config_dict)

        return initial_params


          
    def optimize(self):
        """Run the optimization process."""
        configspace = self.get_config()
        
        # Set up distributed computing
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(min(self.n_workers, torch.cuda.device_count())))), 
            threads_per_worker=1, 
            memory_limit="90GB", 
            device_memory_limit=0.9
        )
        client = Client(cluster)
        logger.info(f"Client: {client}")
        
        # Define objectives
        objectives = self.evaluate_tasks  
        
        # Configure optimization scenario
        scenario = Scenario(
            configspace, 
            output_directory=Path(self.output_path),
            deterministic=True, 
            n_trials=self.n_trials, 
            objectives=objectives,
            min_budget=self.min_budget, 
            max_budget=self.max_budget,
            seed=0,
        )
        # Set up optimization algorithm and intensifier
        multi_objective_algorithm = ParEGO(scenario)
        intensifier = Hyperband(scenario, incumbent_selection="highest_budget")
        # Handle warm start if run history exists
        if self.load_run_history != None:
            runhistory = RunHistory()
            runhistory.update_from_json(self.load_run_history, configspace)
            initial_design=HyperparameterOptimizationFacade.get_initial_design(
                scenario, 
                n_configs=0,  # use the default initial design (random init)
                additional_configs=None,  # Use the configurations previously evaluated as initial design
            )
            smac = HyperparameterOptimizationFacade(
                scenario,
                self.objective,  # We pass the target function here
                overwrite=False,  # Overrides any previous results that are found that are inconsistent with the meta-data
                intensifier=intensifier,
                initial_design=initial_design,
                multi_objective_algorithm=multi_objective_algorithm,
                logging_level=0,
                dask_client=client
            )
            
            # Load previous evaluations
            for (trial_key, trial_value) in runhistory.items():
                trial_info = TrialInfo(
                    config=runhistory.get_config(trial_key.config_id),
                    instance=trial_key.instance,
                    seed=trial_key.seed,
                )
                smac.tell(trial_info, trial_value) 
        else:
            # Generate initial configurations
            init_trials = self.get_initial_params()
            configurations = [Configuration(configspace, trial, allow_inactive_with_values=True) for trial in init_trials]
            logger.info(f"Number of initial configurations: {len(configurations)}")
            initial_design=HyperparameterOptimizationFacade.get_initial_design(
                scenario,
                n_configs=self.random_init_points,
                additional_configs=configurations
            )
      
            smac = HyperparameterOptimizationFacade(
                scenario,
                self.objective,
                overwrite=False ,
                intensifier=intensifier,
                initial_design=initial_design,
                multi_objective_algorithm=multi_objective_algorithm,
                logging_level=0,
                dask_client=client
            )
        # Run optimization
        incumbent = smac.optimize()
        # self.statistics_manager.final_report()
        return incumbent
    
    def eval_output(self):
        result = self.evaluator_instance.evaluate(self.output_path)
        return result
         
    
    def eval_config(self, config, config_id=0):
        """Evaluate a specific configuration."""
        logger.info(f"start eval, config is : {config}")
        result = {}
       
        slices, output_scales = self.generate_genotype(config)
        get_figure(slices, os.path.join(self.output_path, str(config_id)+".png"), output_scales)
        logger.info(f"current genotype : {slices}")
        merge_utils = MergeUtils(
            base_model=self.base_model,
            merging_models=None,
            merging_method=None,
            slices=slices,
            model_storage_path=self.output_path,
            in_memory=self.in_memory_evaluate,
            output_scales=output_scales
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
            
            if self.in_memory_evaluate:
                del out_tensors
            del merge_utils._out_tensors
            del merge_utils
            self.evaluator_instance._destroy_llm()
            gc.collect()    
        except Exception as e:
            logger.info(traceback.format_exc())
            try:
                self.evaluator_instance._destroy_llm()
                gc.collect()
                if self.in_memory_evaluate:
                    del out_tensors
                del merge_utils._out_tensors
                del merge_utils
                
            except:
                logger.error("fail to eval and clean fail")
                logger.error(traceback.format_exc())
            result[self.evaluate_tasks[0]]={}
            result[self.evaluate_tasks[0]]['score'] = 0
        return result
        
    def merge(self):
        study = self.optimize()

        # Save study as a JSON file
        with open(os.path.join(self.output_path, "study.json"), "w") as f:
            json.dump([dict(conf) for conf in study], f, indent=2)
        
if __name__ == "__main__":
    pass

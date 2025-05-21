import gc
import json
import os
import time
import traceback
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

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


class DisMerge(MergeStrategy):
    def __init__(self, config):
        super().__init__(config) 
        logger.info(f"config : {self.config}")
        self.models = self.config["models"]
        self.base_model = self.config["base_model"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.random_init_points = self.config.get("random_init_points", 0)
        self.repeat = self.config["repeat"]
        self.fix_layers = self.config["fix_layers"]
        self.num_hidden_layers = self.config["num_hidden_layers"]
        self.layer_granularity = self.config.get("layer_granularity", 1)
        self.remain_base = self.config.get("remain_base", True)
        self.block_permutation = True if self.repeat==1 else False
        self.candidate_layers = int(self.num_hidden_layers - self.fix_layers * 2) // self.layer_granularity
        self.block_candidate_per_layer = len(self.models) * self.repeat * self.layer_granularity
        self.max_layers = self.config.get("layers", 50)
        self.float_block_selection = self.config.get("float_block_selection", False)
        self.block_selection_threshold = self.config.get("block_selection_threshold", 0.5)
        if self.block_permutation:
            # Generate all permutations
            self.block_permutations = self.get_block_permutations()
    
        self.evaluate_tasks = [task['task'] for task in self.config.get('evaluation', {}).get('tasks', [])]
        self.n_workers = config.get("n_workers", 1)
        self.n_trials = config.get("n_trials")
        self.min_budget = config.get("min_budget", 100)
        self.max_budget = config.get("max_budget", 1000)
        self.total_budget = config.get("total_budget")
        self.eta = config.get("eta", 3)
                
        self.merging_method = config.get("merging_method", "passthrough")
        assert (self.merging_method == "passthrough"), (
            "Merging method must be set to 'passthrough'."
        )
        self.output_path = self.config.get("output_path", None)
        
        self.in_memory_evaluate = (
            self.config.get('evaluation', {}).get('in_memory', False)
        )
        self.evaluator_class = (
            evaluator_classes['inmemory_evaluate'] 
            if self.in_memory_evaluate 
            else evaluator_classes['ondisk_evaluate']
        )
        self.evaluator_instance = self.evaluator_class(self.config)

    
    def generate_fix_layers(self, start, end, model):
        logger.info(f"generating fix layers from {start} to {end}")
        slices_list = []
        for layer_idx in range(start, end): 
            slices_list.append({
                "sources": [
                    {
                        "model": model,
                        "layer_range": [layer_idx, layer_idx+1],  
                    }
                ],
                "merging_method": {self.merging_method: {"scale": [{"value": 1.0}]}}
            })
        logger.info("add fix layers")
        logger.info(slices_list)
        return slices_list


    def get_block_permutations(self):
        block_candidates = self.models * self.repeat * self.layer_granularity
        n = len(block_candidates)
        indices = list(range(n))
        
        def is_valid_perm(perm):
            for i in range(n):
                for j in range(i + 1, n):
                    if block_candidates[indices[i]] == block_candidates[indices[j]] and perm[i] > perm[j]:
                        return False
            return True
        
        result = []
        for p in permutations(indices):
            if is_valid_perm(p):
                result.append(list(p))
        
        return result


    def generate_genotype(self, config):
        slices = []
        output_scales = []

        def get_blocks(layer_permutation_idx, block_selection):
            selected_ordered_blocks = []
            candidates = self.models * self.repeat * self.layer_granularity
            if len(self.block_permutations)>1:
                layer_permutation = self.block_permutations[int(layer_permutation_idx)]
            else:
                layer_permutation = list(range(len(self.models)))
            for idx in layer_permutation:
                if block_selection[idx] == "1":
                    if self.layer_granularity == 1:
                        selected_ordered_blocks.append((0, candidates[idx]))
                    else:
                        selected_ordered_blocks.append((idx // self.layer_granularity, candidates[idx]))
            if len(selected_ordered_blocks) == 0 and self.remain_base:
                selected_ordered_blocks.append((0, self.base_model))
            return selected_ordered_blocks

        start_slice_list = self.generate_fix_layers(0, self.fix_layers, self.models[0])
        end_slice_list = self.generate_fix_layers(
            self.num_hidden_layers - self.fix_layers, self.num_hidden_layers, self.models[0]
        )

        logger.info("add start layer")
        if start_slice_list:
            slices.extend(start_slice_list)
            output_scales.extend([1] * len(start_slice_list))

        global_idx = 0
        for layer_idx in range(self.candidate_layers):
            layer_scale = config.get(f'scale_{layer_idx}', 1)
            if len(self.block_permutations)>1:
                layer_permutation = config.get(f'layer_{layer_idx}_permutation_selection', 1)
            else:
                layer_permutation = 0
            layer_block_list = []
            for block_idx in range(self.block_candidate_per_layer):
                if self.float_block_selection:
                    layer_block_list.append('1' if config.get(f"layer_{layer_idx}_block_{block_idx}_selection", 0) > self.block_selection_threshold else '0')
                else:
                    layer_block_list.append(config.get(f"layer_{layer_idx}_block_{block_idx}_selection", '0'))
            selected_blocks = get_blocks(layer_permutation, layer_block_list)

            logger.info(f"Selected blocks for layer_idx {layer_idx}: {selected_blocks}")
            for block_idx, item in enumerate(selected_blocks):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(f"Invalid block format: {item}")
                layer_id, selected_block = item
                block_scale = layer_scale if block_idx == len(selected_blocks) - 1 else 1
                slice_dict = {
                    "sources": [
                        {
                            "model": selected_block,
                            "layer_range": [
                                self.fix_layers + global_idx + layer_id,
                                self.fix_layers + global_idx + 1 + layer_id,
                            ],
                        }
                    ],
                    "merging_method": {self.merging_method: {"scale": [{"value": 1.0}]}},
                }
                slices.append(slice_dict)
                output_scales.append(block_scale)
            global_idx += self.layer_granularity

        if end_slice_list:
            slices.extend(end_slice_list)
            output_scales.extend([1] * len(end_slice_list))
        return slices, output_scales
            
    
    def get_initial_params(self):
        initial_params = []
        for block_idx in range(self.block_candidate_per_layer // self.layer_granularity):  # Iterate over each block index
            config_dict = {}
            
            for layer in range(self.candidate_layers):
                # Set a default scale value
                config_dict[f'scale_{layer}'] = 1.0
                
                if len(self.block_permutations)>1:
                    config_dict[f'layer_{layer}_permutation_selection'] = str(0)
                
                # Set all blocks to "0", except the chosen block
                for block in range(self.block_candidate_per_layer):
                    if self.layer_granularity == 1:
                        if self.float_block_selection:
                            config_dict[f'layer_{layer}_block_{block}_selection'] = 1 if (block // self.layer_granularity) == block_idx else 0
                        else:
                            config_dict[f'layer_{layer}_block_{block}_selection'] = "1" if (block // self.layer_granularity) == block_idx else "0"
                    else:
                        if self.float_block_selection:
                            config_dict[f'layer_{layer}_block_{block}_selection'] = 1 if (block % self.layer_granularity) == block_idx else 0
                        else:
                            config_dict[f'layer_{layer}_block_{block}_selection'] = "1" if (block % self.layer_granularity) == block_idx else "0"
            initial_params.append(config_dict)
        
        return initial_params
    
            
    def objective(self, config, seed, budget):
        budget = int(budget)
        result = {}
        slices, output_scales = self.generate_genotype(config) 
        logger.info(f"current genotype : {slices}")
        layer_len = len(slices)   
        if layer_len > self.max_layers:
            logger.info(f"layer {layer_len} too large, set reward to 0")
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1
            
            return result[self.evaluate_tasks[0]]
        
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
                eval_result = self.evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer, budget)  
            else:
                eval_result = self.evaluator_instance.evaluate(self.output_path, budget)  
            # Manually release memory here to address SMAC's behavior of creating a new instance each time.
            # Todo: This causes the initialization of vllm each time, which can be time-consuming
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
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1
            try:
                del out_tensors
                del merge_utils._out_tensors
                del merge_utils
                self.evaluator_instance._destroy_llm()
                self.evaluator_instance._clean_inner_model()
                gc.collect()  
            except:
                logger.error("fail to eval and clean fail")
                logger.error(traceback.format_exc())
        return result[self.evaluate_tasks[0]]

         
    def get_config(self):
        # Build Configuration Space which defines all parameters and their ranges.
        cs = ConfigurationSpace()
        config_list = []
        
        for i in range(self.candidate_layers):
            config_list.append(Float(f'scale_{i}', (0, 2), default=1))
            if len(self.block_permutations) > 1:
                config_list.append(Categorical(f"layer_{i}_permutation_selection", items=[str(idx) for idx in range(len(self.block_permutations))]))
            for j in range(self.block_candidate_per_layer):
                if self.float_block_selection:
                    config_list.append(Float(f"layer_{i}_block_{j}_selection", (0, 1), default=0.5))
                else:
                    config_list.append(Categorical(f"layer_{i}_block_{j}_selection", items=["0", "1"]))
        cs.add_hyperparameters(config_list)
        return cs
           
    def optimize(self):
        configspace = self.get_config()
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
       
        scenario = Scenario(
            configspace, 
            output_directory=Path(self.output_path),
            deterministic=True, 
            n_trials=self.n_trials, 
            min_budget=self.min_budget, 
            max_budget=self.max_budget
        )
        
        intensifier = MultiFidelityFacade.get_intensifier(scenario=scenario, eta=self.eta)
        # warm start
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
                initial_design=initial_design,
                logging_level=0,
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
            init_trials = self.get_initial_params()
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
        
    
    def eval_config(self, config, config_id=0):
        logger.info(f"start eval, config is : {config}")
        result = {}
        sample = {}
        slices, output_scales = self.generate_genotype(config)
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
                result, sample = self.evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer)   
            else:
                result, sample = self.evaluator_instance.evaluate(self.output_path)  
            
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
                del out_tensors
                del merge_utils._out_tensors
                del merge_utils
                
            except:
                logger.error("fail to eval and clean fail")
                logger.error(traceback.format_exc())
            result[self.evaluate_tasks[0]]={}
            result[self.evaluate_tasks[0]]['score'] = 0
        return result, sample
        
    def merge(self):
        study = self.optimize()
        
if __name__ == "__main__":
    pass

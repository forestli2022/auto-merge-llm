import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
import transformers

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks

from utils import logger

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

def eval_model(model, tasks, model_args=None, task_manager=None, **kwargs):
    logger.info("Starting evaluation...")
    logger.info(model)
    logger.info(model_args)
    logger.info([{k: v for k, v in task.items() if k != 'metric'} for task in tasks])
    logger.info(task_manager)
    logger.info(kwargs)

    results = None
    while results is None:
        try:
            results = lm_eval.evaluator.simple_evaluate(
                model=model,
                model_args=model_args,
                tasks=[{k: v for k, v in task.items() if k not in ["metric", "limit"]} for task in tasks], 
                log_samples=True, # for debug
                verbosity="DEBUG",
                task_manager=task_manager,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Exception during evaluation: {e}")
            results = None
            # Sleep for a while
            time.sleep(10)

    for task in tasks:
        results["results"][task['task']]['score'] = results["results"][task['task']][task['metric']]
    logger.info(f"Evaluation results: {results['results']}")
    return results["results"] 
    
def evaluate_model(merged_path, tasks, num_fewshot, limit, vllm, 
                   batch_size, device, task_manager, torch_dtype="float32"):
    try:
        model_args = {
            "pretrained": merged_path,
            "dtype": torch_dtype,
        }
        if vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = 32
            model_args["max_model_len"] = 4096 #2048 mbpp #1024 
        else:
            model_args["use_cache"] = True

        res = eval_model(
            "vllm" if vllm else "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            device="cuda" if vllm else device, # vLLM only supports CUDA
            task_manager=task_manager,
        )
        return res
    finally:
        pass


class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True

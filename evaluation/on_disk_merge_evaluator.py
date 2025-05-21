import gc
import torch

from utils import logger
from .base_evaluator import MergeActorBase
from .evaluate_helper import evaluate_model


class OnDiskMergeEvaluator(MergeActorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def evaluate(self, merged_path, sample_size=None):
        gc.collect()
        torch.cuda.empty_cache()
        
        if not merged_path:
            logger.error("Can not find Model merge path")
            return {"score": None, "results": None}

        return evaluate_model(
            merged_path,
            self.tasks,
            num_fewshot=self.num_fewshot,
            limit=sample_size if sample_size != None else self.limit,
            vllm=self.vllm,
            batch_size=self.batch_size,
            device=self.device,
            task_manager=self.task_manager,
            torch_dtype=self.torch_dtype
        )
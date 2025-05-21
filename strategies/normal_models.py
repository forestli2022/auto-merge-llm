from evaluation import evaluator_classes
from utils import logger
from .base_strategy import MergeStrategy
from .merge_utils import MergeUtils

class NormalModelsMerge(MergeStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.base_model = self.config.get("base_model", None)
        self.merging_models = self.config.get("models", None)
        self.merging_method = self.config.get("merging_method", None)
        self.output_path = self.config.get("output_path", None)
        self.evaluate_tasks = self.config.get('evaluation', {}).get('tasks', [])
        self.in_memory_evaluate = self.config.get('evaluation', {}).get('in_memory', False)
        self.evaluator_class = (
            evaluator_classes['inmemory_evaluate'] 
            if self.in_memory_evaluate 
            else evaluator_classes['ondisk_evaluate']
        )
        self.evaluator_instance = self.evaluator_class(self.config) 
        
        self.merge_utils = MergeUtils(
            base_model=self.base_model, 
            merging_models=self.merging_models, 
            merging_method=self.merging_method, 
            slices=None, 
            model_storage_path=self.output_path,
            in_memory=self.in_memory_evaluate, 
            device=None
        )
               
    def merge(self):
        self.merge_utils.merge_models_v1()
        if self.in_memory_evaluate:
            out_tensors = self.merge_utils.out_tensors
            output_config = self.merge_utils.output_config
            aligned_tokenizer = self.merge_utils.aligned_tokenizer
            self.evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer)
        else:
            self.evaluator_instance.evaluate(self.output_path)
      
if __name__ == "__main__":
    pass


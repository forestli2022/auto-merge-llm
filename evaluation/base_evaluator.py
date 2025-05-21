from lm_eval.tasks import TaskManager

from utils import logger 
from custom_models import GetAnswer

class MergeActorBase:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.vllm = self.evaluation_config.get('vllm', False)
        self.batch_size = self.evaluation_config.get('batch_size', 16)
        self.num_fewshot = self.evaluation_config.get('num_fewshot', None)
        self.limit = self.evaluation_config.get('limit', None)
        self.torch_dtype = self.evaluation_config.get('torch_dtype', "float16")
        self.enforce_eager = self.evaluation_config.get('enforce_eager', True)
        self.device = self.evaluation_config.get('device', None)
        include_path = self.evaluation_config.get('include_path', None)
        self.task_manager = TaskManager(include_path=include_path)
        self.tasks = self.evaluation_config.get('tasks', [])
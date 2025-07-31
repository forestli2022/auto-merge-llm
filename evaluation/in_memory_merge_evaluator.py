import os
import gc
import ray
import tempfile
import traceback
import contextlib
import time
import torch
import torch.cuda
import torch.distributed
import transformers
from transformers.utils import is_flash_attn_2_available
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from vllm.utils import is_cpu
import contextlib
import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface

from utils import logger
from .base_evaluator import MergeActorBase
from .evaluate_helper import evaluate_model, eval_model, NoInit

try:
    import vllm
except ImportError:
    vllm = None

# Configure CUDA to synchronize kernel launches for easier debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Map string dtype names to PyTorch types  
dtype_mapping = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
}

class InMemoryMergeEvaluator(MergeActorBase):
    """Evaluates merged models in-memory without saving to disk."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None 
        self.inner_model = None
        self.model_storage_path = self.config['output_path']
        self.arch_info = None
        self.max_model_len = None
        self.is_flash_attn_2_available = True

    def _is_arch_diff(self, current_arch_info):
        different = False
        if self.arch_info is None:
            different = True
        if self.arch_info is not None:
            for key in current_arch_info.to_diff_dict():
                if key in ["architectures", "model_type"]:
                    continue
                elif key in ["use_cache", "torch_dtype"]:
                    continue
                elif key.endswith("_token_id"):
                    setattr(
                        self.arch_info, 
                        key, 
                        getattr(current_arch_info, key, None)
                    )
                    continue

                if (getattr(current_arch_info, key) !=
                        getattr(self.arch_info, key, None)):
                    logger.warning(
                        "Config key %s changed, reinitializing model",
                        key
                    )
                    different = True
                    break
        return different
    
    def _get_max_model_len(self, arch_config):
        if (
            seq_len := getattr(arch_config, "max_position_embeddings", None)
        ) is not None:
            self.max_model_len = seq_len
        if (window_sz := getattr(arch_config, "sliding_window", None)) is not None:
            self.max_model_len = min(self.max_model_len or 1024, window_sz)
        if self.max_model_len and self.max_model_len > 8192:
            self.max_model_len = 8192
            logger.warn(f"Clipping sequence length to {self.max_model_len}")
    
    def _instance_vllm(self, tokenizer):
        if self.model_storage_path and not os.path.exists(self.model_storage_path):
            os.makedirs(self.model_storage_path) 
        with tempfile.TemporaryDirectory(
                dir=self.model_storage_path, prefix="vllm"
            ) as tempdir:
            self.inner_model.save_pretrained(
                tempdir, safe_serialization=True, out_shard_size=1_000_000_000_000
            )
            tokenizer.save_pretrained(tempdir)
            self._clean_inner_model()
            torch.cuda.empty_cache()
            # https://github.com/vllm-project/vllm/issues/188
            self._show_memory_usage("before instance vllm")
            self.model = lm_eval.models.vllm_causallms.VLLM(
                pretrained=tempdir,
                enforce_eager=self.enforce_eager,
                batch_size=self.batch_size, 
                max_model_len= 2048, # Fixed context window
                gpu_memory_utilization=1,
                dtype=self.torch_dtype,
                device="cuda",
                trust_remote_code=True,
            )


    def _show_memory_usage(self, usage_before=''):
        logger.info(f"cuda memory {usage_before}: {torch.cuda.memory_allocated()//1024//1024}MB")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"  --> after gc {usage_before}: {torch.cuda.memory_allocated()//1024//1024}MB")
    
    
    def _clean_inner_model(self):
        try:
            if not hasattr(self, 'inner_model') or self.inner_model is None:
                return
                      
            # Move to CPU and delete
            self.inner_model.cpu()
            del self.inner_model
            self.inner_model = None
            
            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in _clean_inner_model: {e}")
    
    def _check_memory_released(self, threshold):
        allocated_memory = torch.cuda.memory_allocated()
        logger.info(f"Allocated memory: {allocated_memory / (1024 ** 2)} MB")  
        return allocated_memory < threshold 
    
    def _destroy_llm(self, max_retries=3, retry_delay=3):
        """Properly destroy VLLM model to free GPU memory."""
        # It only works with vllm 0.4.0.post1, not sure how to handle other versions. :(
        # https://github.com/vllm-project/vllm/issues/1908
        self._show_memory_usage("start destroying former llm")
        for attempt in range(max_retries):
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'llm_engine'):
                    # Original approach
                    logger.info("Destroying LLM with expected structure 0")
                    del self.model.model.llm_engine.model_executor.driver_worker
                    del self.model.model.llm_engine.model_executor
                    del self.model.model
                elif hasattr(self.model, 'llm_engine'):
                    # Alternative structure
                    logger.info("Destroying LLM with expected structure 1")
                    del self.model.llm_engine.model_executor.driver_worker
                    del self.model.llm_engine
                elif hasattr(self.model, 'engine'):
                    # Another possible structure
                    del self.model.engine
                else:
                    # If none of the above structures match
                    logger.warning("Unknown model structure. Attempting to delete the entire model.")
                    logger.info(f"Model type: {type(self.model)}")
                   
                for _ in range(3):
                    gc.collect()

                self._show_memory_usage("after destroying former llm")
                if self._check_memory_released(threshold=10 * 1024 * 1024 * 1024):  # 30 GB in bytes
                    logger.info("Memory successfully released below threshold.")
                    break
                else:
                    logger.warning(f"Memory usage exceeds threshold. Attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)  # Wait before retrying   
            except Exception as e:
                logger.error(f"Error occurred while destroying LLM: {e}")
                logger.info(traceback.print_exc())
                break

    def _get_gpu_memory_usage(self):
        """
        Get current GPU memory usage
        Returns:
            dict: Dictionary containing memory usage for each GPU
        """
        gpu_memory = {}
        try:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)  # Convert to GB
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024 * 1024)    # Convert to GB
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': f'{allocated:.2f}GB',
                    'reserved': f'{reserved:.2f}GB'
                }
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {str(e)}")
            return None
        return gpu_memory

    def _init_model_with_retry(self, current_arch_info, tokenizer, max_retries=3, retry_delay=3):
        """Initialize model with retry mechanism"""
        for attempt in range(max_retries):
            try:
                self._init_model(current_arch_info, tokenizer)
                logger.info(f"Model initialization successful on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Error during model initialization (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(traceback.format_exc())
                
                # Log GPU memory usage on failure
                gpu_memory = self._get_gpu_memory_usage()
                if gpu_memory:
                    logger.error("Current GPU memory usage:")
                    for gpu, memory in gpu_memory.items():
                        logger.error(f"{gpu}: Allocated: {memory['allocated']}, Reserved: {memory['reserved']}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Try to clean up before retry
                    try:
                        if self.vllm and self.model is not None:
                            self._destroy_llm()
                        if hasattr(self, 'inner_model'):
                            self._clean_inner_model()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        # Log memory usage after cleanup
                        gpu_memory = self._get_gpu_memory_usage()
                        if gpu_memory:
                            logger.info("GPU memory usage after cleanup:")
                            for gpu, memory in gpu_memory.items():
                                logger.info(f"{gpu}: Allocated: {memory['allocated']}, Reserved: {memory['reserved']}")
                                
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup before retry: {str(cleanup_error)}")
                else:
                    logger.error("Max retries reached. Model initialization failed.")
                    logger.error(f"failed init model {current_arch_info}")
                    # Log final memory state
                    gpu_memory = self._get_gpu_memory_usage()
                    if gpu_memory:
                        logger.error("Final GPU memory state:")
                        for gpu, memory in gpu_memory.items():
                            logger.error(f"{gpu}: Allocated: {memory['allocated']}, Reserved: {memory['reserved']}")
                    raise

    def _init_model(self, current_arch_info, tokenizer):
        """
        Core model initialization logic extracted from original _init_model
        """
        is_arch_diff = self._is_arch_diff(current_arch_info)
        if is_arch_diff:
            if self.vllm and self.model is not None:
                try:
                    self._destroy_llm()
                except:
                    logger.info(traceback.format_exc())
            
            with NoInit():
                selected_dtype = dtype_mapping.get(self.torch_dtype, torch.float32)
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": selected_dtype,
                }
                #if self.is_flash_attn_2_available:
                #    model_kwargs["attn_implementation"] = "flash_attention_2"
                
                self.inner_model = transformers.AutoModelForCausalLM.from_config(
                    current_arch_info,
                    **model_kwargs,
                )
                
                if selected_dtype == torch.bfloat16:
                    self.inner_model = self.inner_model.bfloat16()
                elif selected_dtype == torch.float16:
                    self.inner_model = self.inner_model.half()
                elif selected_dtype == torch.float32:
                    self.inner_model = self.inner_model.float()
                
                self.inner_model = self.inner_model.eval().requires_grad_(False)
                
                if self.vllm:
                    self._get_max_model_len(current_arch_info)
                    self._instance_vllm(tokenizer)
                    try:
                        self._clean_inner_model()
                    except:
                        logger.info("inner model already be deleted")
                else:
                    self.model = lm_eval.models.huggingface.HFLM(
                        pretrained=self.inner_model, 
                        tokenizer=tokenizer, 
                        device=self.device
                    )
                    self._clean_inner_model()
            
            self._show_memory_usage("after init model")
            self.arch_info = current_arch_info
                        
    def evaluate(self, out_tensors, current_arch_info, tokenizer, sample_size=None):
        sample_size = int(sample_size) if sample_size!= None and sample_size > 1 else sample_size
        gc.collect()
        torch.cuda.empty_cache()
        gpu_memory = self._get_gpu_memory_usage()
        if gpu_memory:
            logger.error("Start eval: Current GPU memory usage:")
            for gpu, memory in gpu_memory.items():
                logger.error(f"{gpu}: Allocated: {memory['allocated']}, Reserved: {memory['reserved']}")
        
        self._init_model_with_retry(current_arch_info, tokenizer)
        model = self.model.model
        if self.vllm and isinstance(model, vllm.LLM):
            assert (
                model.llm_engine.parallel_config.world_size == 1
            ), "Must be single GPU"
            # In the new version of vllm, an abstract class `Executor` has been added.
            worker = model.llm_engine.model_executor.driver_worker
            model = worker.model_runner.model
        
        param_dict = dict(model.named_parameters())
        
        stacked_mapping = {
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }
        
        for tensor_name, value in out_tensors.items():   
            if "rotary_emb.inv_freq" in tensor_name:
                continue
            if tensor_name in param_dict:
                if tensor_name in ["model.embed_tokens.weight", "lm_head.weight"]:
                    param_tensor = param_dict[tensor_name].data
                    value_tensor = value 
                    padded_value = torch.zeros_like(param_tensor)
                    padded_value[:value.shape[0], :] = value_tensor
                    param_tensor.copy_(padded_value, non_blocking=True)
                else:
                    param_dict[tensor_name].data.copy_(value, non_blocking=True)
            elif self.vllm:
                stacked = False
                for needle, (replacement, shard_id) in stacked_mapping.items():
                    if needle in tensor_name:
                        target = tensor_name.replace(needle, replacement)
                        param = param_dict[target]
                        weight_loader = param.weight_loader
                        weight_loader(param, value, shard_id)
                        stacked = True
                        break

                if not stacked:
                    raise ValueError(f"Unknown parameter {tensor_name}")
            else:
                raise ValueError(f"Unknown parameter {tensor_name}")

            del value
        
        return eval_model(
            self.model,
            self.tasks,
            num_fewshot=self.num_fewshot,
            limit=sample_size if sample_size != None else self.limit,
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            device=self.device
        )

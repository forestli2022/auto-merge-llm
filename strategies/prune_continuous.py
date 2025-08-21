import gc
import json
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# Local application imports
from evaluation import evaluator_classes
from evaluation.entropy_trainer import EntropyTrainer
from loader.tensor_loader import TensorLoader
from loader.tensor_writer import TensorWriter
from methods.weighted_task_vectors_grad import WeightedTaskVectorsGrad
from tokenizer.build_tokenizer_embed import align_tokenizers_and_embeddings_v1
from utils import logger
from utils.utils import get_model_storage_path
from .base_strategy import MergeStrategy


SUPPORTED_METHOD_PARAM_MAPS = {
    "linear": ["weights"],
    "task_arithmetic": ["scaling_coefficient"],
    "ties": ["param_value_mask_rate", "scaling_coefficient"],
    "slerp": ["slerp_t" ]
}


class PruneContinuous(MergeStrategy):
    def __init__(self, config):
        super().__init__(config) 
        logger.info(f"config : {self.config}")
        
        # Extract configuration parameters
        self.models = self.config["models"]
        self.base_model = self.config["base_model"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.num_hidden_layers = self.config["num_hidden_layers"]
        self.candidate_layers = int(self.num_hidden_layers)
        self.candidates_per_layer = len(self.models)
        self.max_layers = self.config.get("layers", 40)
        self.remove_layers = self.num_hidden_layers - self.max_layers

        self.dtype = self.config.get("dtype", "float16")
        self.evaluate_tasks = [task['task'] for task in self.config.get('evaluation', {}).get('tasks', [])]
        self.batch_size = self.config.get('evaluation', {}).get('batch_size', 8)
        self.limit = self.config.get('evaluation', {}).get('limit', 100)
        self.in_memory_evaluate = self.config.get("in_memory", False)
        self.evaluator_class = (
            evaluator_classes['inmemory_evaluate']
            if self.in_memory_evaluate 
            else evaluator_classes['ondisk_evaluate']
        )

        
        # Optimization parameters
        self.n_trials = config.get("n_trials")
        self.reg_warmup_steps = config.get("reg_warmup_steps", 300)
        self.seed = config.get("seed", 0)
        
        # Merging configuration    
        self.output_path = self.config.get("output_path", None)
        
        # Construct merging parameters. Currently only allow weighted task vector. One merging weight per layer
        self.merging_weights = nn.ParameterDict({
            f"layer_{i}_merging_weights": nn.Parameter(torch.full((self.candidates_per_layer,), 0.3))
            for i in range(self.num_hidden_layers)
        })

        # Construct pruning gates, 1 gate per layer
        self.merging_weights["prune_weights"] = nn.Parameter(
            0.0 + 0.01 * torch.randn(self.num_hidden_layers)
        )

     # ---------- checkpoint utils ----------
    def _ckpt_dir(self) -> Path:
        base = Path(self.output_path) if self.output_path else Path.cwd()
        return base / "checkpoints"

    def _ckpt_path(self, epoch: int) -> Path:
        return self._ckpt_dir() / f"merging_weights_epoch_{epoch:04d}.pt"

    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer | None = None) -> None:
        ckpt_dir = self._ckpt_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save only ParameterDict weights (and optionally optimizer)
        state = {
            "epoch": int(epoch),
            "merging_weights": {k: v.detach().cpu() for k, v in self.merging_weights.items()},
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()

        path = self._ckpt_path(epoch)
        torch.save(state, path)
        logger.info(f"[ckpt] saved: {path}")

        # Also write/refresh a 'latest.pt' pointer for convenience
        latest = ckpt_dir / "latest.pt"
        try:
            torch.save(state, latest)
        except Exception:
            pass

    def _find_latest_checkpoint(self) -> Path | None:
        ckpt_dir = self._ckpt_dir()
        if not ckpt_dir.exists():
            return None
        files = sorted(ckpt_dir.glob("merging_weights_epoch_*.pt"))
        return files[-1] if files else None

    def _load_checkpoint_if_available(self, optimizer: torch.optim.Optimizer | None = None) -> int:
        """
        Returns the starting epoch index (0-based) to resume from.
        If self.load_run_history is:
          - truthy bool -> load the latest checkpoint in default dir
          - str path    -> load that specific file
        """
        if not self.load_run_history:
            return 0

        if isinstance(self.load_run_history, str):
            ckpt_path = Path(self.load_run_history)
            if not ckpt_path.exists():
                logger.warning(f"[ckpt] resume path not found: {ckpt_path}")
                return 0
        else:
            ckpt_path = self._find_latest_checkpoint()
            if ckpt_path is None:
                logger.info("[ckpt] no checkpoint found to resume; starting fresh")
                return 0

        try:
            state = torch.load(ckpt_path, map_location="cpu")
            # Load ParameterDict weights
            with torch.no_grad():
                for k, v in state.get("merging_weights", {}).items():
                    if k in self.merging_weights:
                        self.merging_weights[k].copy_(v)
            # Optionally restore optimizer
            if optimizer is not None and "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])

            epoch = int(state.get("epoch", 0))
            start_epoch = max(0, epoch)  # resume AFTER this epoch
            logger.info(f"[ckpt] loaded {ckpt_path} (epoch={epoch}) -> start_epoch={start_epoch}")
            return start_epoch
        except Exception as e:
            logger.warning(f"[ckpt] failed to load {ckpt_path}: {e}")
            return 0
        
    def merge(self):
        """
        Merges and prune the models, using continuous method
        """
        logger.info("Starting continuous merging/pruning")

        # Build optimizer
        optimizer = torch.optim.Adam(self.merging_weights.parameters(), lr=1e-3)

        # Build entropy trainer
        ev = EntropyTrainer(device="cuda" if torch.cuda.is_available() else "cpu", dtype=self.dtype)

        # Load checkpoint
        start_epoch = self._load_checkpoint_if_available(optimizer)

        # Overall optimization loop
        for epoch in range(start_epoch, self.n_trials):
            logger.info(f"Epoch {epoch+1}/{self.n_trials}:")

            for task in self.evaluate_tasks:
                logger.info(f"Training on task {task}...")

                bs = self.batch_size if task != "mmlu" else self.batch_size // 4
                num_batches = ev.num_batches(task, batch_size=bs, limit=self.limit, split="validation", seed=self.seed)

                for batch in tqdm(range(num_batches), desc="batch"):
                    # Zero grad for this batch
                    optimizer.zero_grad(set_to_none=True)

                    # Initialize continuous merger
                    tau = 1.0
                    continuous_merger = ContinuousMerger(
                        merging_weights=self.merging_weights,
                        temperature=tau,
                        num_total_layers=self.num_hidden_layers,
                        num_result_layers=self.max_layers,
                        base_model=self.base_model,
                        merging_models=self.models,
                        model_storage_path=self.output_path,
                    )

                    # Build merged model for training
                    continuous_merger.build_merged_model_for_training()

                    # Batch process calibration data
                    entropy = ev.batch_loss_by_index(
                        out_tensors=continuous_merger.out_tensors,
                        arch_info=continuous_merger.output_config,
                        tokenizer=continuous_merger.aligned_tokenizer,
                        task=task,
                        batch_idx=batch,
                        batch_size=bs,
                        limit=self.limit,
                        split="validation",
                        seed=self.seed,
                    )

                    # Loss calculation
                    reg_weight = self._linear_ramp(epoch)
                    loss = entropy + self._gate_regularizer(continuous_merger.gates, k=self.max_layers)
                    logger.info(f"Epoch {epoch}, Task {task}, Batch {batch+1}/{num_batches}, Regression Weight: {reg_weight}, Loss: {loss.item()}")

                    # Backprop
                    loss.backward()
                    optimizer.step()

                    del continuous_merger

            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                self._save_checkpoint(epoch+1, optimizer)
        
        logger.info("Finished merging/pruning")
        self.evaluate()

    def _linear_ramp(self, epoch):
        return min(1.0, epoch / max(1, self.reg_warmup_steps))


    def _gate_regularizer(
        self,
        gates: nn.Parameter,
        k: int
    ):
        # # Constraint: there needs to be exactly k high gates
        # logger.info(f"Gate mean: {gates.mean()}, k: {k}, gate numel: {gates.numel()}")
        # loss_count = (gates.sum() - k )**2 / gates.numel()
        # logger.info(f"Loss count: {loss_count.item()}")

        # # Constraint: pushes gates to either 1 or 0 (Bernoulli entropy)
        # eps = 1e-8
        # ge = gates.clamp(eps, 1 - eps)
        # H = -(ge * torch.log2(ge) + (1 - ge) * torch.log2(1 - ge))
        # loss_bin = 5 * H.mean() # FIXME
        # logger.info(f"Loss bin: {loss_bin.item()}")

        return 0.0

    def evaluate(self):
        logger.info(f"Starting evaluation")

        # Initialize continuous merger
        tau = 1.0
        continuous_merger = ContinuousMerger(
            merging_weights=self.merging_weights,
            temperature=tau,
            num_total_layers=self.num_hidden_layers,
            num_result_layers=self.max_layers,
            base_model=self.base_model,
            merging_models=self.models,
            model_storage_path=self.output_path,
        )

        # Build merged model for training
        continuous_merger.build_merged_model_for_inference(num_layers_to_keep=self.max_layers)
        if not self.in_memory_evaluate:
            continuous_merger.save_pretrained()

        try:
            # Construct new config with full eval tasks
            self.config['evaluation']['tasks'] = self.config['evaluation']["final_evaluation_tasks"]
            self.config['evaluation']['limit'] = None
            full_evaluator_instance = self.evaluator_class(self.config)

            # Evaluate
            if self.in_memory_evaluate:
                out_tensors = continuous_merger.out_tensors
                output_config = continuous_merger.output_config
                aligned_tokenizer = continuous_merger.aligned_tokenizer
                result = full_evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer)
            else:
                result = full_evaluator_instance.evaluate(self.output_path)

            # Cleanup
            if self.in_memory_evaluate:
                del out_tensors
                full_evaluator_instance._destroy_llm()
            del continuous_merger._out_tensors
            del continuous_merger
            gc.collect()
        except Exception as e:
            logger.info(traceback.format_exc())
            try:
                gc.collect()
                if self.in_memory_evaluate:
                    del out_tensors
                    full_evaluator_instance._destroy_llm()
                del continuous_merger._out_tensors
                del continuous_merger
            except:
                logger.error("fail to eval and clean fail")
                logger.error(traceback.format_exc())
            result["error"] = str(e)

        logger.info(f"Results: {result}")
        with open(os.path.join(self.output_path, f"evaluation result.json"), "w") as f:
            json.dump(result, f, indent=2)


class ContinuousMerger:
    def __init__(
        self, 
        merging_weights: nn.ParameterDict,
        temperature: float,
        num_total_layers: int,
        num_result_layers: int,
        base_model: str, 
        merging_models: list[str], 
        model_storage_path: str, 
        device: str = "cpu"
    ):
        self.merging_weights = merging_weights
        self.num_total_layers = num_total_layers
        self.num_result_layers = num_result_layers
        self.base_model = base_model
        self.merging_models = merging_models
        self.model_storage_path = model_storage_path
        self.device = device

        # Alphas and gates
        self._alphas = [
            torch.sigmoid(self.merging_weights[f"layer_{i}_merging_weights"]).to("cpu")
            for i in range(self.num_total_layers)
        ]
        # STE trick to generate hard gates
        self.g_soft = torch.sigmoid(self.merging_weights["prune_weights"] / temperature)
        _, idx = torch.topk(self.g_soft, k=num_result_layers)
        g_hard = torch.zeros_like(self.g_soft)
        for i in idx:
            g_hard[i] = 1.0
        self._gates = self.g_soft + (g_hard - self.g_soft).detach()


        # Weighted task vector merging operation
        self._wtv_grad = WeightedTaskVectorsGrad()

        # Caches / configs
        self.base_model_cache: TensorLoader = None
        self.merging_model_caches: TensorLoader = {}
        self.arch_config: Dict[str, str] = {}
        self.base_tokenizer_config: Dict = {}
        self.merging_models_tokenizer_config: Dict[str, str] = {}

        # Outputs
        self._out_tensors: Dict[str, torch.Tensor] = {}
        self._aligned_tokenizer = None
        self._output_config = None

    # Public props
    @property
    def out_tensors(self) -> Dict[str, torch.Tensor]:
        return self._out_tensors

    @property
    def output_config(self):
        return self._output_config

    @property
    def aligned_tokenizer(self):
        return self._aligned_tokenizer

    @property
    def alphas(self):
        return self._alphas
    
    @property
    def gates(self):
        return self._gates

    def build_merged_model_for_training(self) -> None:
        """
        1. load caches/configs
        2. align tokenizer & embeddings
        3. merge with per-layer weights
        4. merge postweights
        5. update output_config
        """
        logger.info(f"[ContinuousMerger] Start building merged model for training with alphas = {[alpha.data for alpha in self._alphas]}, soft gates = {self.g_soft}")

        self._load_caches_and_configs()
        self._build_tokenizer_and_embed()

        for layer_idx in range(self.num_total_layers):
            self._merge_layer(layer_idx)
        
        self._merge_postweights()
        self._update_output_config()

        logger.info("Finished building merged model for training")

    def build_merged_model_for_inference(self, num_layers_to_keep) -> None:
        logger.info(f"[ContinuousMerger] Start building merged model for inference with alphas = {[alpha.data for alpha in self._alphas]}, gates = {self._gates.data}")

        self._load_caches_and_configs()
        self._build_tokenizer_and_embed()

        _, keep_indices = torch.topk(self._gates, k=num_layers_to_keep)
        keep_indices = keep_indices.sort().values.tolist()

        for layer_idx in range(self.num_total_layers):
            if layer_idx in keep_indices:
                self._merge_layer(layer_idx)
        
        self._merge_postweights()
        self._update_output_config(num_layers_kept=num_layers_to_keep)
        self._correct_out_tensor_names(keep_indices)

        logger.info("[ContinuousMerger] Finished building merged model for inference.")

    def save_pretrained(self) -> None:
        """Save merged tensors + config + tokenizer under model_storage_path."""
        os.makedirs(self.model_storage_path, exist_ok=True)
        writer = TensorWriter(self.model_storage_path)

        for name, t in self._out_tensors.items():
            writer.save_tensor(name=name, tensor=t.detach().to("cpu"))
        writer.finalize()

        self._output_config.save_pretrained(self.model_storage_path)
        self._aligned_tokenizer.save_pretrained(self.model_storage_path)
        logger.info(f"[ContinuousMerger] saved to: {self.model_storage_path}")

    # Internals
    def _load_tokenizer_and_config(self, model: str) -> Dict:
        CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")
        out = {}
        try:
            local = get_model_storage_path(model)
            out["tokenizer"] = AutoTokenizer.from_pretrained(local)
            out["config"] = AutoConfig.from_pretrained(local)
        except Exception as e:
            # If model not in cache, download it from hf
            out["tokenizer"] = AutoTokenizer.from_pretrained(model, CACHE_DIR=CACHE_DIR)
            out["config"] = AutoConfig.from_pretrained(model, CACHE_DIR=CACHE_DIR)
        return out

    def _load_caches_and_configs(self) -> None:
        # Donor models
        self.merging_model_caches = {
            model: TensorLoader(model_name=model, lazy_unpickle=False, device=self.device)
            for model in self.merging_models
        }
        self.merging_models_tokenizer_config = {
            model: self._load_tokenizer_and_config(model) for model in self.merging_models
        }

        # Base model
        self.base_model_cache = TensorLoader(model_name=self.base_model, lazy_unpickle=False, device=self.device)
        self.base_tokenizer_config = self._load_tokenizer_and_config(self.base_model)
        self._output_config = self.base_tokenizer_config["config"]

        self.arch_config = self.base_model_cache.tensor_paths

    def _find_single_weight_name(self, pattern: str) -> str:
        rx = re.compile(pattern, re.IGNORECASE)
        names = [k for k in self.arch_config.keys() if rx.search(k)]
        assert len(names) == 1, f"Expected one match for '{pattern}', found {len(names)}"
        return names[0]

    def _get_matches_weight_names(self, filter_name: str, match_layer: bool = False) -> List[str]:
        if match_layer:
            rx = re.compile(rf"model\.layers\.{filter_name}\..+")
            return [k for k in self.arch_config.keys() if rx.search(k) and "rotary_emb.inv_freq" not in k]
        rx = re.compile(filter_name, re.IGNORECASE)
        return [k for k in self.arch_config.keys() if rx.search(k)]

    def _build_tokenizer_and_embed(self) -> None:
        # Locate embed names
        input_embed_name = self._find_single_weight_name("embed")
        try: 
            output_embed_name = self._find_single_weight_name("lm_head")
        except:
            output_embed_name = self._find_single_weight_name("embed_tokens")

        # base embeds
        base_in = self.base_model_cache.get_tensor(input_embed_name)
        base_out = self.base_model_cache.get_tensor(output_embed_name)

        # donor embeds
        donor_in = {m: self.merging_model_caches[m].get_tensor(input_embed_name) for m in self.merging_models}
        donor_out = {m: self.merging_model_caches[m].get_tensor(output_embed_name) for m in self.merging_models}

        # align
        aligned = align_tokenizers_and_embeddings_v1(
            [base_in, base_out],
            self.base_tokenizer_config,
            [donor_in, donor_out],
            self.merging_models_tokenizer_config,
        )

        self._aligned_tokenizer = self.base_tokenizer_config["tokenizer"]

        # Merge embeddings
        last_alpha = self._alphas[self.num_total_layers - 1]  # tensor on CPU, grads alive
        alpha_list = [last_alpha[i] for i in range(last_alpha.numel())]

        base_in_aligned = aligned["base"]["input_aligned_embed"]
        base_out_aligned = aligned["base"]["output_aligned_embed"]
        donor_in_aligned = [aligned[m]["input_aligned_embed"] for m in self.merging_models]
        donor_out_aligned = [aligned[m]["output_aligned_embed"] for m in self.merging_models]

        merged_in = self._merge_tensor(
            base_tensor=base_in_aligned,
            donor_tensors=donor_in_aligned,
            alpha_list=alpha_list,
            tensor_name=input_embed_name,
        )
        merged_out = self._merge_tensor(
            base_tensor=base_out_aligned,
            donor_tensors=donor_out_aligned,
            alpha_list=alpha_list,
            tensor_name=output_embed_name,
        )

        self._out_tensors[input_embed_name] = merged_in
        self._out_tensors[output_embed_name] = merged_out

    def _merge_tensor(
        self,
        base_tensor: torch.Tensor,
        donor_tensors: List[torch.Tensor],
        alpha_list: nn.Parameter,
        tensor_name: str,
    ) -> torch.Tensor:
        merged_cpu = self._wtv_grad.merge_tensor(
            base_tensor=base_tensor,
            tensors_to_merge=donor_tensors,
            method_params=alpha_list,
            tensor_name=tensor_name
        )
        return merged_cpu.to(self.device)

    def _merge_layer(self, layer_idx: int) -> None:
        weight_names = self._get_matches_weight_names(str(layer_idx), match_layer=True)
        alpha_list = self._alphas[layer_idx]

        for weight_name in weight_names:
            base_tensor = self.base_model_cache.get_tensor(weight_name)
            donor_tensors = [self.merging_model_caches[m].get_tensor(weight_name) for m in self.merging_models]
            self._out_tensors[weight_name] = self._merge_tensor(
                base_tensor=base_tensor,
                donor_tensors=donor_tensors,
                alpha_list=alpha_list,
                tensor_name=weight_name
            ) * self._gates[layer_idx]

    def _merge_postweights(self) -> None:
        post = self._get_matches_weight_names(r"model\.norm\.weight")
        if not post:
            return
        assert len(post) == 1, f"Expected one post-norm weight, found: {post}"
        name = post[0]

        alpha_list = self._alphas[self.num_total_layers - 1]

        base_t = self.base_model_cache.get_tensor(name)
        donor_ts = [self.merging_model_caches[m].get_tensor(name) for m in self.merging_models]
        self._out_tensors[name] = self._merge_tensor(
            base_tensor=base_t,
            donor_tensors=donor_ts,
            alpha_list=alpha_list,
            tensor_name=name,
        )

    def _update_output_config(self, num_layers_kept=None) -> None:
        if num_layers_kept is not None:
            self._output_config.update({"num_hidden_layers": int(num_layers_kept)})
        else:
            self._output_config.update({"num_hidden_layers": int(self.num_total_layers)})
        self._output_config.update({"vocab_size": int(len(self._aligned_tokenizer.get_vocab()))})

    def _correct_out_tensor_names(self, remaining_indices):
        logger.info(f"[ContinuousMerger] Remaining indices after folding: {remaining_indices}")
        # Need to fix the tensor names and remove redundant tensors after collapsing
        index_map = {} # map to track the new indices of layers
        new_idx = 0
        # e.g. if we have 3 layers and the second layer is collapsed, we will have a mapping like {0: 0, 2: 1}
        for old_idx in remaining_indices:
            index_map[old_idx] = new_idx
            new_idx += 1
        # Update the _output_tensors with the retained layer indices
        layer_pat = re.compile(r"model\.layers\.(\d+)\.")
        new_tensors = {}
        for name, tensor in self._out_tensors.items():
            m = layer_pat.search(name)
            if m:
                old_idx = int(m.group(1))
                if old_idx in index_map.keys():
                    new_idx = index_map[old_idx]
                    new_name = name.replace(f"layers.{old_idx}.", f"layers.{new_idx}.")
                    new_tensors[new_name] = tensor
            else:
                new_tensors[name] = tensor     # embeddings, lm_head, etc.
        self._out_tensors = new_tensors
        self._output_config.num_hidden_layers = len(set(index_map.values()))

        # Log the final number of layers, which layers are collapsed
        logger.info(f"[ContinuousMerger] Final number of layers: {self._output_config.num_hidden_layers}")
        logger.info(f"[ContinuousMerger] Layers retained: {set(index_map.keys())}")
import gc
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# Local application imports
from evaluation import evaluator_classes
from evaluation.accuracy_trainer import AccuracyTrainer
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

# ----------------------------
# Parameter-space layer embedder (per-block random projection)
# ----------------------------

class ParameterLayerEmbedder:
    """
    Build fixed-size vector per layer from parameters. Uses per-block random projection.
    """
    def __init__(self, per_block_dim: int = 32, seed: int = 0, device: str = "cpu"):
        self.per_block_dim = per_block_dim
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def _get_layer_weight_names(self, keys: List[str], idx: int) -> List[str]:
        layer_regx = re.compile(r"model\.layers\.(\d+)\.")
        skip_regx = re.compile(r"rotary_emb\.inv_freq|^model\.embed_tokens|lm_head")
        names = []
        for k in keys:
            if skip_regx.search(k):
                continue
            m = layer_regx.search(k)
            if m and int(m.group(1)) == idx:
                names.append(k)
        names.sort()
        return names

    def _flatten_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().reshape(-1)
        x = x - x.mean()
        rms = x.norm() / math.sqrt(max(1, x.numel()))
        return x / (rms + 1e-8)

    def _random_mat(self, n: int, d: int, gen: torch.Generator, device="cpu"):
        return torch.randn(n, d, generator=gen, device=device) / math.sqrt(d)

    @torch.no_grad()
    def embed_model(
        self,
        model_cache: TensorLoader,
        num_layers: int,
    ) -> np.ndarray:
        """
        Embed each layer's parameters into a low-dim vector via per-block RP.
        Returns: [num_layers, F]
        """
        arch_keys = list(model_cache.tensor_paths.keys())
        layer_vecs: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            weight_names = self._get_layer_weight_names(arch_keys, layer_idx)
            feats: List[torch.Tensor] = []
            for wi, name in enumerate(weight_names):
                logger.info(f"[Embedder] Computing projection for {name}")
                tensor = model_cache.get_tensor(name)
                tensor = self._flatten_norm(tensor).to(self.device)

                # Seed per (layer, block) for determinism
                generator = torch.Generator(device=self.device)
                generator.manual_seed(397 * (layer_idx + 1) + 7919 * (wi + 1))

                # Random projection for this block
                G = self._random_mat(tensor.numel(), self.per_block_dim, generator, self.device)
                z = tensor @ G  # [per_block_dim]
                feats.append(z)

            if not feats:
                # Safety if layer has no tensors (unlikely)
                feats.append(torch.zeros(self.per_block_dim, device=self.device))

            v = torch.cat(feats, dim=0)  # [sum_blocks * per_block_dim]
            # Normalize for GMM stability
            v = (v - v.mean()) / (v.std() + 1e-6)
            v = v / (v.norm() + 1e-8)
            layer_vecs.append(v.detach().cpu().numpy())

        return np.stack(layer_vecs, axis=0)


# ----------------------------
# Main strategy
# ----------------------------

class FoldGMM(MergeStrategy):
    def __init__(self, config):
        super().__init__(config)
        logger.info(f"config : {self.config}")

        # Extract configuration parameters
        self.models = self.config["models"]
        self.base_model = self.config["base_model"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.num_hidden_layers = int(self.config["num_hidden_layers"])
        self.candidate_layers = self.num_hidden_layers
        self.candidates_per_layer = len(self.models)
        self.max_layers = int(self.config.get("layers", 40))
        self.remove_layers = self.num_hidden_layers - self.max_layers
        self.per_block_dim = int(self.config.get("per_block_dim", 32))

        self.dtype = self.config.get("dtype", "float16")
        self.evaluate_tasks = [task['task'] for task in self.config.get('evaluation', {}).get('tasks', [])]
        self.batch_size = int(self.config.get('evaluation', {}).get('batch_size', 8))
        self.limit = self.config.get('evaluation', {}).get('limit', 100)
        self.in_memory_evaluate = bool(self.config.get("in_memory", False))
        self.evaluator_class = (
            evaluator_classes['inmemory_evaluate']
            if self.in_memory_evaluate
            else evaluator_classes['ondisk_evaluate']
        )

        # Optimization parameters
        self.n_trials = int(config.get("n_trials"))
        self.seed = int(config.get("seed", 0))

        # Merging configuration
        self.output_path = self.config.get("output_path", None)

        # ---------- GMM on base model → adjacency clusters ----------
        logger.info("Fitting GMM on base model layer embeddings to form adjacency clusters...")
        base_cache = TensorLoader(model_name=self.base_model, lazy_unpickle=False, device="cpu")
        embedder = ParameterLayerEmbedder(per_block_dim=self.per_block_dim, seed=self.seed, device="cpu")

        base_layer_vecs = embedder.embed_model(
            model_cache=base_cache,
            num_layers=self.num_hidden_layers,
        )  # [L, F]
        logger.info(f"Base layer vectors: {base_layer_vecs.shape}")

        k = min(self.max_layers, self.num_hidden_layers)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',
            n_init=5,
            reg_covar=1e-6,
            random_state=self.seed
        ).fit(base_layer_vecs)

        R = gmm.predict_proba(base_layer_vecs)  # [L, k]
        self.clusters = self._greedy_adjacent_groups(R, target_groups=self.max_layers)  # list[list[int]]
        logger.info(f"[clusters] {self.clusters}")

        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, "base_adjacent_groups.json"), "w") as f:
                json.dump(self.clusters, f, indent=2)

        # ---------- Initialize learnable weights AFTER clusters are known ----------
        # Alphas: per layer (0..L-1), one weight per expert model.
        # Betas:  ONLY for NON-LEADER layers (those that collapse into their cluster leader).
        alpha_init = -1.0
        beta_init = -2.0
        self.merging_weights = nn.ParameterDict({
            **{
                f"layer_{i}_merging_weights": nn.Parameter(torch.full((self.candidates_per_layer,), alpha_init))
                for i in range(self.num_hidden_layers)
            },
            **{
                f"layer_{li}_collapse_weight": nn.Parameter(torch.tensor(beta_init))
                for cluster in self.clusters for li in cluster[1:]  # only non-leaders
            }
        })

    # ---------- checkpoint utils ----------
    def _ckpt_dir(self) -> Path:
        base = Path(self.output_path) if self.output_path else Path.cwd()
        return base / "checkpoints"

    def _ckpt_path(self, epoch: int) -> Path:
        return self._ckpt_dir() / f"merging_weights_epoch_{epoch:04d}.pt"

    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer | None = None) -> None:
        ckpt_dir = self._ckpt_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": int(epoch),
            "merging_weights": {k: v.detach().cpu() for k, v in self.merging_weights.items()},
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()

        path = self._ckpt_path(epoch)
        torch.save(state, path)
        logger.info(f"[ckpt] saved: {path}")

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
            with torch.no_grad():
                for k, v in state.get("merging_weights", {}).items():
                    if k in self.merging_weights:
                        self.merging_weights[k].copy_(v)
            if optimizer is not None and "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])

            epoch = int(state.get("epoch", 0))
            start_epoch = max(0, epoch)
            logger.info(f"[ckpt] loaded {ckpt_path} (epoch={epoch}) -> start_epoch={start_epoch}")
            return start_epoch
        except Exception as e:
            logger.warning(f"[ckpt] failed to load {ckpt_path}: {e}")
            return 0

    # ---------- clustering helpers ----------
    def _js_divergence(self, p, q, eps=1e-12):
        p = np.clip(p, eps, 1.0); p = p / p.sum()
        q = np.clip(q, eps, 1.0); q = q / q.sum()
        m = 0.5 * (p + q)
        def _kl(a, b): return float((a * (np.log(a) - np.log(b))).sum())
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    def _greedy_adjacent_groups(self, R: np.ndarray, target_groups: int) -> List[List[int]]:
        assert R.ndim == 2
        L, K = R.shape
        assert 1 <= target_groups <= L

        groups = [[i] for i in range(L)]
        sizes  = [1] * L
        G = [R[i].copy() for i in range(L)]   # group responsibility distributions

        while len(groups) > target_groups:
            dists = [self._js_divergence(G[i], G[i+1]) for i in range(len(G)-1)]
            j = int(np.argmin(dists))
            sA, sB = sizes[j], sizes[j+1]
            G[j] = (sA * G[j] + sB * G[j+1]) / (sA + sB)
            sizes[j] = sA + sB
            groups[j] = groups[j] + groups[j+1]
            del G[j+1]; del sizes[j+1]; del groups[j+1]

        return groups

    # ---------- training / evaluation ----------
    def merge(self):
        """
        Train alphas/betas by minimizing entropy loss on tasks while folding clusters.
        """
        logger.info("Start folding/merging...")

        optimizer = torch.optim.Adam(self.merging_weights.parameters(), lr=1e-3)
        ev = EntropyTrainer(device="cuda" if torch.cuda.is_available() else "cpu", dtype=self.dtype)
        # ev = AccuracyTrainer(device="cuda" if torch.cuda.is_available() else "cpu", dtype=self.dtype)

        start_epoch = self._load_checkpoint_if_available(optimizer)

        for epoch in range(start_epoch, self.n_trials):
            logger.info(f"Epoch {epoch+1}/{self.n_trials}:")
            for task in self.evaluate_tasks:
                logger.info(f"Training on task {task}...")

                bs = self.batch_size if task != "mmlu" else max(1, self.batch_size // 4)
                num_batches = ev.num_batches(task, batch_size=bs, limit=self.limit, split="validation", seed=self.seed)

                for batch in tqdm(range(num_batches), desc="batch"):
                    optimizer.zero_grad(set_to_none=True)

                    continuous_merger = ContinuousMerger(
                        clusters=self.clusters,
                        merging_weights=self.merging_weights,
                        num_total_layers=self.num_hidden_layers,
                        num_result_layers=self.max_layers,
                        base_model=self.base_model,
                        merging_models=self.models,
                        model_storage_path=self.output_path,
                    )

                    continuous_merger.build_merged_model()

                    loss = ev.batch_loss_by_index(
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

                    logger.info(f"Epoch {epoch}, Task {task}, Batch {batch+1}/{num_batches}, Loss: {loss.item()}")

                    loss.backward()
                    optimizer.step()

                    del continuous_merger

            if (epoch + 1) % 2 == 0:
                self._save_checkpoint(epoch+1, optimizer)

        logger.info("Finished merging/pruning")
        self.evaluate()

    def evaluate(self):
        logger.info(f"Starting evaluation")

        continuous_merger = ContinuousMerger(
            clusters=self.clusters,
            merging_weights=self.merging_weights,
            num_total_layers=self.num_hidden_layers,
            num_result_layers=self.max_layers,
            base_model=self.base_model,
            merging_models=self.models,
            model_storage_path=self.output_path,
        )

        continuous_merger.build_merged_model()
        if not self.in_memory_evaluate:
            continuous_merger.save_pretrained()

        try:
            self.config['evaluation']['tasks'] = self.config['evaluation']["final_evaluation_tasks"]
            self.config['evaluation']['limit'] = None
            full_evaluator_instance = self.evaluator_class(self.config)

            if self.in_memory_evaluate:
                out_tensors = continuous_merger.out_tensors
                output_config = continuous_merger.output_config
                aligned_tokenizer = continuous_merger.aligned_tokenizer
                result = full_evaluator_instance.evaluate(out_tensors, output_config, aligned_tokenizer)
            else:
                result = full_evaluator_instance.evaluate(self.output_path)

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
            result = {"error": str(e)}

        logger.info(f"Results: {result}")
        with open(os.path.join(self.output_path, f"evaluation result.json"), "w") as f:
            json.dump(result, f, indent=2)


# ----------------------------
# ContinuousMerger: merges across experts (alphas), then collapses clusters (betas)
# ----------------------------

class ContinuousMerger:
    def __init__(
        self,
        clusters: List[List[int]],
        merging_weights: nn.ParameterDict,
        num_total_layers: int,
        num_result_layers: int,
        base_model: str,
        merging_models: List[str],
        model_storage_path: str,
        device: str = "cpu"
    ):
        self.clusters = [sorted(c) for c in clusters]
        self.merging_weights = merging_weights
        self.num_total_layers = num_total_layers
        self.num_result_layers = num_result_layers
        self.base_model = base_model
        self.merging_models = merging_models
        self.model_storage_path = model_storage_path
        self.device = device

        # Alphas (per-layer per-expert), passed through sigmoid in accessor
        self._alphas = [
            torch.sigmoid(self.merging_weights[f"layer_{i}_merging_weights"]).to("cpu")
            for i in range(self.num_total_layers)
        ]
        # Betas: only exist for non-leader layers; map layer_idx -> beta scalar
        self._beta: Dict[int, torch.Tensor] = {}
        for c in self.clusters:
            for li in c[1:]:
                key = f"layer_{li}_collapse_weight"
                if key in self.merging_weights:
                    self._beta[li] = torch.sigmoid(self.merging_weights[key]).to("cpu")

        logger.info(f"[ContinuousMerger] Constructed alphas: {[alpha.data for alpha in self._alphas]}")
        logger.info(f"[ContinuousMerger] Constructed betas: {self._beta}")

        # Weighted task vector merging operation
        self._wtv_grad = WeightedTaskVectorsGrad()

        # Caches / configs
        self.base_model_cache: TensorLoader = None
        self.merging_model_caches: Dict[str, TensorLoader] = {}
        self.arch_config: Dict[str, str] = {}
        self.base_tokenizer_config: Dict = {}
        self.merging_models_tokenizer_config: Dict[str, Dict] = {}

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

    def build_merged_model(self) -> None:
        """
        1. load caches/configs
        2. align tokenizer & embeddings
        3. merge with per-layer weights (alphas across experts)
        4. collapse layers inside each cluster (betas)
        5. merge postweights
        6. update output_config & rename layers to be contiguous
        """
        logger.info(f"[ContinuousMerger] Build start")

        self._load_caches_and_configs()
        self._build_tokenizer_and_embed()

        # 1) Merge each layer across experts using alphas
        for layer_idx in range(self.num_total_layers):
            self._merge_layer(layer_idx)

        # 2) Collapse: fold non-leaders into leader using betas
        for cluster in self.clusters:
            self._collapse_layers(cluster)

        # 3) Postweights and config
        self._merge_postweights()
        self._update_output_config(num_layers_kept=len(self.clusters))

        # 4) Rename: keep leaders, compact indices
        leaders_sorted = sorted([c[0] for c in self.clusters])
        self._correct_out_tensor_names(leaders_sorted)

        logger.info("[ContinuousMerger] Build done")

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
        except Exception:
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

        # Merge embeddings using alphas of the last layer (your original convention)
        last_alpha = self._alphas[self.num_total_layers - 1]
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
        alpha_list: List[torch.Tensor],
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
                alpha_list=[alpha_list[i] for i in range(alpha_list.numel())],
                tensor_name=weight_name
            )

    def _collapse_layers(self, cluster: List[int]) -> None:
        """
        Merge every non-leader layer into the first (leader) layer in the cluster,
        using WeightedTaskVectorsGrad with betas as weights.
        """
        if not cluster or len(cluster) == 1:
            return

        leader = int(cluster[0])
        others = [int(x) for x in cluster[1:]]

        # All weight names for leader (we'll map to donor names in 'others')
        leader_weight_names = self._get_matches_weight_names(str(leader), match_layer=True)

        for lw_name in leader_weight_names:
            # Start from the already expert-merged leader tensor
            base_t = self._out_tensors[lw_name]

            donor_ts: List[torch.Tensor] = []
            betas: List[torch.Tensor] = []

            for li in others:
                # Same tensor, but from layer 'li'
                ow_name = lw_name.replace(f"layers.{leader}.", f"layers.{li}.")
                if ow_name not in self._out_tensors:
                    # Skip tensors that didn't exist or were missing (defensive)
                    continue
                donor_ts.append(self._out_tensors[ow_name])
                # Only non-leaders have beta params; we created them in FoldGaussian.__init__
                beta = self._beta.get(li, torch.tensor(0.0))
                betas.append(beta.squeeze())

            if donor_ts:
                merged = self._wtv_grad.merge_tensor(
                    base_tensor=base_t,
                    tensors_to_merge=donor_ts,
                    method_params=betas,   # one beta per donor layer
                    tensor_name=lw_name
                )
                self._out_tensors[lw_name] = merged

        # Delete tensors of donor layers now that they've been folded
        layer_pat = re.compile(r"model\.layers\.(\d+)\.")
        to_delete = []
        for name in list(self._out_tensors.keys()):
            m = layer_pat.search(name)
            if m and int(m.group(1)) in others:
                to_delete.append(name)
        for name in to_delete:
            del self._out_tensors[name]

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
            alpha_list=[alpha_list[i] for i in range(alpha_list.numel())],
            tensor_name=name,
        )

    def _update_output_config(self, num_layers_kept=None) -> None:
        if num_layers_kept is not None:
            self._output_config.update({"num_hidden_layers": int(num_layers_kept)})
        else:
            self._output_config.update({"num_hidden_layers": int(self.num_total_layers)})
        self._output_config.update({"vocab_size": int(len(self._aligned_tokenizer.get_vocab()))})

    def _correct_out_tensor_names(self, remaining_indices: List[int]):
        logger.info(f"[ContinuousMerger] Remaining leader indices before renumbering: {remaining_indices}")
        # Map old indices -> new compact indices [0..K-1] in ascending order of 'remaining_indices'
        index_map = {}
        for new_idx, old_idx in enumerate(sorted(remaining_indices)):
            index_map[old_idx] = new_idx

        layer_pat = re.compile(r"model\.layers\.(\d+)\.")
        new_tensors = {}
        for name, tensor in self._out_tensors.items():
            m = layer_pat.search(name)
            if m:
                old_idx = int(m.group(1))
                if old_idx in index_map:
                    new_name = name.replace(f"layers.{old_idx}.", f"layers.{index_map[old_idx]}.")
                    new_tensors[new_name] = tensor
                # else: it's a folded layer; it should already have been deleted
            else:
                new_tensors[name] = tensor  # embeddings, lm_head, etc.

        self._out_tensors = new_tensors
        self._output_config.num_hidden_layers = len(remaining_indices)

        logger.info(f"[ContinuousMerger] Final number of layers: {self._output_config.num_hidden_layers}")
        logger.info(f"[ContinuousMerger] Layers retained (leaders old idx): {set(remaining_indices)}")

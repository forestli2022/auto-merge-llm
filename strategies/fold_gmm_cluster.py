from utils.probe import probe as cuda_probe, empty_cache as cuda_empty_cache

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
from evaluation.entropy_trainer import EntropyTrainer  # use entropy for training (differentiable)
# from evaluation.accuracy_trainer import AccuracyTrainer  # optional for eval only
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
    "slerp": ["slerp_t"],
}

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
                tensor = model_cache.get_tensor(name)
                tensor = self._flatten_norm(tensor).to(self.device)

                # Seed per (layer, block) for determinism
                generator = torch.Generator(device=self.device)
                generator.manual_seed(397 * (layer_idx + 1) + 7919 * (wi + 1))

                # Random projection for this block
                G = self._random_mat(tensor.numel(), self.per_block_dim, generator, self.device)
                z = tensor @ G  # [per_block_dim]
                feats.append(z)

                logger.info(f"[embed] layer {layer_idx} block {wi} name={name} shape={tuple(z.shape)} isnan={torch.isnan(z).any().item()}")

            if not feats:
                feats.append(torch.zeros(self.per_block_dim, device=self.device))

            v = torch.cat(feats, dim=0)  # [sum_blocks * per_block_dim]
            v = (v - v.mean()) / (v.std() + 1e-6)
            v = v / (v.norm() + 1e-8)
            layer_vecs.append(v.detach().cpu().numpy())

        return np.stack(layer_vecs, axis=0)


class FoldGMMCluster(MergeStrategy):
    def __init__(self, config):
        super().__init__(config)
        logger.info(f"config : {self.config}")

        # Extract configuration parameters
        self.models = list(self.config["models"])  # donor models
        self.base_model = self.config["base_model"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.num_hidden_layers = int(self.config["num_hidden_layers"])
        self.max_layers = int(self.config.get("layers", 40))
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

        # ---------- Try to reuse a prior cluster init ----------
        self.resp_by_model: Dict[str, np.ndarray] = {}
        self.clusters_by_model: Dict[str, List[List[int]]] = {}
        self.cluster_specs: List[Dict] = []
        self.merging_weights = nn.ParameterDict()

        reused = self._load_cluster_init_if_available()
        if not reused:
            logger.info("Fitting GMM on ALL models' layer embeddings to form per-model adjacency clusters...")

            base_cache = TensorLoader(model_name=self.base_model, lazy_unpickle=False, device="cpu")
            donor_caches = {m: TensorLoader(model_name=m, lazy_unpickle=False, device="cpu") for m in self.models}
            embedder = ParameterLayerEmbedder(per_block_dim=self.per_block_dim, seed=self.seed, device="cpu")

            # Per-model layer embeddings
            layer_vecs_by_model: Dict[str, np.ndarray] = {}
            layer_vecs_by_model[self.base_model] = embedder.embed_model(base_cache, self.num_hidden_layers)  # [L, F]
            for m in self.models:
                layer_vecs_by_model[m] = embedder.embed_model(donor_caches[m], self.num_hidden_layers)       # [L, F]

            # Fit one GMM on the pooled matrix [(M+1)*L, F]
            pooled = np.concatenate([layer_vecs_by_model[self.base_model]] +
                                    [layer_vecs_by_model[m] for m in self.models], axis=0)  # [(M+1)*L, F]
            logger.info(f"[embed] pooled shape: {pooled.shape}")
            k = min(self.max_layers, self.num_hidden_layers)
            gmm = GaussianMixture(
                n_components=k, covariance_type='diag', n_init=5, reg_covar=1e-6, random_state=self.seed
            ).fit(pooled)

            # Per-model adjacency clusters using GMM responsibilities for THAT model's layers only
            self.clusters_by_model = {}
            self.resp_by_model = {}
            for model_name, layer_vecs in layer_vecs_by_model.items():
                R = gmm.predict_proba(layer_vecs)  # [L, k]
                self.resp_by_model[model_name] = R
                clusters = self._greedy_adjacent_groups(R, target_groups=k)  # list[list[int]]
                self.clusters_by_model[model_name] = clusters
                logger.info(f"[clusters] {model_name}: {clusters}")

            if self.output_path:
                os.makedirs(self.output_path, exist_ok=True)
                with open(os.path.join(self.output_path, "adjacent_groups_by_model.json"), "w") as f:
                    json.dump(self.clusters_by_model, f, indent=2)

            # ---------- Build cluster specs (donor ordering per cluster) ----------
            self.cluster_specs = []
            base_clusters = self.clusters_by_model[self.base_model]
            for j in range(k):
                leader = int(base_clusters[j][0])
                donor_refs: List[Tuple[str, int]] = []
                for li in base_clusters[j][1:]:
                    donor_refs.append((self.base_model, int(li)))
                for m in self.models:
                    for li in self.clusters_by_model[m][j]:
                        donor_refs.append((m, int(li)))
                spec = {"cluster_idx": j, "leader": leader, "donor_refs": donor_refs}
                self.cluster_specs.append(spec)

            logger.info(f"Cluster specs: {self.cluster_specs}")

            # ---------- Initialize learnable weights (RESP-distance init) ----------
            self.merging_weights = nn.ParameterDict()
            self._init_cluster_weights_from_resp_distance(tau=0.15, target_sum=None, distance="js")

            # Embedding weights: keep your previous “normal” init (zeros ⇒ sigmoid≈0.5)
            if len(self.models) > 0:
                self.merging_weights["embed_weights"] = nn.Parameter(
                    torch.zeros(len(self.models), dtype=torch.float32)
                )

            # Save initialized artifacts so we can skip GMM next time
            self._save_cluster_init()
        else:
            logger.info(f"Cluster specs: {self.cluster_specs}")
            logger.info(f"Weight: {self.merging_weights}")

        logger.info(f"Weight: {self.merging_weights}")

    # ---------- cluster init IO ----------
    def _cluster_init_dir(self) -> Path:
        base = Path(self.output_path) if self.output_path else Path.cwd()
        return base / "cluster_init"

    def _cluster_init_meta(self) -> dict:
        return {
            "base_model": self.base_model,
            "models": list(self.models),
            "num_hidden_layers": int(self.num_hidden_layers),
            "max_layers": int(self.max_layers),
            "per_block_dim": int(self.per_block_dim),
            "seed": int(self.seed),
        }

    def _save_cluster_init(self) -> None:
        d = self._cluster_init_dir()
        d.mkdir(parents=True, exist_ok=True)

        meta = self._cluster_init_meta()
        with open(d / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        with open(d / "clusters_by_model.json", "w") as f:
            json.dump(self.clusters_by_model, f, indent=2)
        with open(d / "cluster_specs.json", "w") as f:
            json.dump(self.cluster_specs, f, indent=2)
        with open(d / "responsibilities.json", "w") as f:
            json.dump({k: v.tolist() for k, v in self.resp_by_model.items()}, f, indent=2)

        mw_state = {k: v.detach().cpu() for k, v in self.merging_weights.items()}
        torch.save(mw_state, d / "merging_weights_init.pt")

        logger.info(f"[cluster-init] saved init artifacts to: {d}")

    def _load_cluster_init_if_available(self) -> bool:
        d = self._cluster_init_dir()
        try:
            meta_path = d / "meta.json"
            spec_path = d / "cluster_specs.json"
            groups_path = d / "clusters_by_model.json"
            resp_path = d / "responsibilities.json"
            mw_path = d / "merging_weights_init.pt"

            if not (meta_path.exists() and spec_path.exists() and groups_path.exists() and mw_path.exists()):
                return False

            with open(meta_path, "r") as f:
                meta = json.load(f)
            ok = (
                meta.get("base_model") == self.base_model and
                meta.get("models") == list(self.models) and
                int(meta.get("num_hidden_layers", -1)) == int(self.num_hidden_layers) and
                int(meta.get("max_layers", -1)) == int(self.max_layers) and
                int(meta.get("per_block_dim", -1)) == int(self.per_block_dim) and
                int(meta.get("seed", -999)) == int(self.seed)
            )
            if not ok:
                logger.info("[cluster-init] init artifacts exist but meta mismatch; recomputing.")
                return False

            with open(groups_path, "r") as f:
                self.clusters_by_model = json.load(f)
            with open(spec_path, "r") as f:
                self.cluster_specs = json.load(f)

            self.resp_by_model = {}
            if resp_path.exists():
                with open(resp_path, "r") as f:
                    raw = json.load(f)
                for k, arr in raw.items():
                    self.resp_by_model[k] = np.array(arr, dtype=np.float64)

            raw_state = torch.load(mw_path, map_location="cpu")
            self.merging_weights = nn.ParameterDict()
            for k, v in raw_state.items():
                self.merging_weights[k] = nn.Parameter(v.clone().detach())

            logger.info(f"[cluster-init] loaded init artifacts from: {d}")
            logger.info(f"[cluster-init] loaded {len(self.cluster_specs)} clusters; "
                        f"merging_weights keys={list(self.merging_weights.keys())}")
            return True

        except Exception as e:
            logger.warning(f"[cluster-init] failed to load init artifacts: {e}")
            return False

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
        L, _ = R.shape
        assert 1 <= target_groups <= L
        groups = [[i] for i in range(L)]
        sizes  = [1] * L
        G = [R[i].copy() for i in range(L)]
        while len(groups) > target_groups:
            dists = [self._js_divergence(G[i], G[i+1]) for i in range(len(G)-1)]
            j = int(np.argmin(dists))
            sA, sB = sizes[j], sizes[j+1]
            G[j] = (sA * G[j] + sB * G[j+1]) / (sA + sB)
            sizes[j] = sA + sB
            groups[j] = groups[j] + groups[j+1]
            del G[j+1]; del sizes[j+1]; del groups[j+1]
        return groups

    # ---------- responsibility-distance initializer ----------
    def _logit(self, p: float, eps: float = 1e-6) -> float:
        p = float(np.clip(p, eps, 1.0 - eps))
        return float(np.log(p / (1.0 - p)))

    def _init_cluster_weights_from_resp_distance(
        self,
        tau: float = 0.15,
        target_sum: float | None = None,
        distance: str = "js"
    ):
        """
        For each cluster j:
          - r_base = responsibilities of base's leader layer
          - For each donor (model, li), compute distance(r_base, r_donor)
          - Convert to affinities via softmax(-dist / tau)
          - Normalize to sum=1 (or 'target_sum' if provided)
          - Set nn.Parameter logits so sigmoid(param) ~= desired weight
        """
        def _dist(p: np.ndarray, q: np.ndarray) -> float:
            if distance == "js":
                return self._js_divergence(p, q)
            elif distance == "l2":
                return float(np.linalg.norm(p - q))
            else:
                raise ValueError(f"unknown distance '{distance}'")

        for spec in self.cluster_specs:
            j = int(spec["cluster_idx"])
            leader = int(spec["leader"])
            donors = list(spec["donor_refs"])

            r_base = self.resp_by_model[self.base_model][leader]  # [k]
            dists = []
            for (m, li) in donors:
                r_d = self.resp_by_model[m][li]  # [k]
                dists.append(_dist(r_base, r_d))

            dists = np.asarray(dists, dtype=np.float64)
            d_shift = dists - dists.min() if np.isfinite(dists).all() else dists
            a = np.exp(-d_shift / max(1e-6, tau))
            if not np.isfinite(a).any() or a.sum() <= 0:
                a = np.ones_like(dists)
            a = a / a.sum()
            if target_sum is not None:
                a = a * float(target_sum)

            init_raw = torch.tensor([self._logit(float(x)) for x in a], dtype=torch.float32)
            self.merging_weights[f"cluster_{j}_weights"] = nn.Parameter(init_raw)

    # ---------- training / evaluation ----------
    def merge(self):
        logger.info("Start folding/merging (cluster-level)...")

        optimizer = torch.optim.Adam(self.merging_weights.parameters(), lr=1e-3)
        ev = EntropyTrainer(device="cuda" if torch.cuda.is_available() else "cpu", dtype=self.dtype)

        start_epoch = self._load_checkpoint_if_available(optimizer)

        for epoch in range(start_epoch, self.n_trials):
            logger.info(f"Epoch {epoch+1}/{self.n_trials}:")
            for task in self.evaluate_tasks:
                logger.info(f"Training on task {task}...")
                bs = self.batch_size
                num_batches = ev.num_batches(task, batch_size=bs, limit=self.limit, split="validation", seed=self.seed)

                for batch in tqdm(range(num_batches), desc="batch"):
                    optimizer.zero_grad(set_to_none=True)

                    continuous_merger = ContinuousMerger(
                        cluster_specs=self.cluster_specs,
                        clusters_by_model=self.clusters_by_model,
                        merging_weights=self.merging_weights,
                        num_total_layers=self.num_hidden_layers,
                        num_result_layers=self.max_layers,
                        base_model=self.base_model,
                        merging_models=self.models,
                        model_storage_path=self.output_path,
                    )
                    continuous_merger.build_merged_model()
                    cuda_probe("after build_merged_model (merged tensors are CPU)", logger=logger)

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

                    # --- NEW: careful per-batch cleanup to keep peak down ---
                    try:
                        del loss
                        del continuous_merger._out_tensors
                        del continuous_merger
                    except Exception:
                        pass
                    gc.collect()
                    cuda_empty_cache()
                    cuda_probe("after batch cleanup", logger=logger)

                if (epoch + 1) % 2 == 0:
                    self._save_checkpoint(epoch+1, optimizer)

        logger.info("Finished merging/pruning")
        self.evaluate()

    def evaluate(self):
        logger.info("Starting evaluation")

        continuous_merger = ContinuousMerger(
            cluster_specs=self.cluster_specs,
            clusters_by_model=self.clusters_by_model,
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
        if self.output_path:
            with open(os.path.join(self.output_path, f"evaluation result.json"), "w") as f:
                json.dump(result, f, indent=2)


# ----------------------------
# ContinuousMerger: cluster-level single-pass merges
# ----------------------------

class ContinuousMerger:
    def __init__(
        self,
        cluster_specs: List[Dict],
        clusters_by_model: Dict[str, List[List[int]]],
        merging_weights: nn.ParameterDict,
        num_total_layers: int,
        num_result_layers: int,
        base_model: str,
        merging_models: List[str],
        model_storage_path: str,
        device: str = "cpu"
    ):
        self.cluster_specs = cluster_specs
        self.clusters_by_model = clusters_by_model
        self.merging_weights = merging_weights
        self.num_total_layers = num_total_layers
        self.num_result_layers = num_result_layers
        self.base_model = base_model
        self.merging_models = merging_models
        self.model_storage_path = model_storage_path
        self.device = device

        # Weighted task vector merging op
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
        3. for each cluster, merge all member layers (base+donor models) into base leader via ONE WTV call
        4. merge postweights
        5. update output_config & rename layers to be contiguous
        """
        logger.info(f"[ContinuousMerger] Build start")

        self._load_caches_and_configs()
        self._build_tokenizer_and_embed()

        # Merge each cluster into base's leader layer index
        for spec in self.cluster_specs:
            self._merge_cluster(spec)

        # Postweights and config
        self._merge_postweights()
        self._update_output_config(num_layers_kept=len(self.cluster_specs))

        # Rename layers: leaders (old base indices) -> 0..K-1
        leaders_sorted = [spec["leader"] for spec in self.cluster_specs]
        self._correct_out_tensor_names(leaders_sorted)

        logger.info("[ContinuousMerger] Build done")

    def save_pretrained(self) -> None:
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
        # Donors
        self.merging_model_caches = {
            model: TensorLoader(model_name=model, lazy_unpickle=False, device=self.device)
            for model in self.merging_models
        }
        self.merging_models_tokenizer_config = {
            model: self._load_tokenizer_and_config(model) for model in self.merging_models
        }

        # Base
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

        # Merge embeddings with a single donor-weight vector (sigmoid)
        if "embed_weights" in self.merging_weights and len(self.merging_models) > 0:
            ew = torch.sigmoid(self.merging_weights["embed_weights"]).to("cpu")
            ew_list = [ew[i] for i in range(ew.numel())]
            base_in_aligned = aligned["base"]["input_aligned_embed"]
            base_out_aligned = aligned["base"]["output_aligned_embed"]
            donor_in_aligned = [aligned[m]["input_aligned_embed"] for m in self.merging_models]
            donor_out_aligned = [aligned[m]["output_aligned_embed"] for m in self.merging_models]

            merged_in = self._merge_tensor(
                base_tensor=base_in_aligned,
                donor_tensors=donor_in_aligned,
                method_params=ew_list,
                tensor_name=input_embed_name,
            )
            merged_out = self._merge_tensor(
                base_tensor=base_out_aligned,
                donor_tensors=donor_out_aligned,
                method_params=ew_list,
                tensor_name=output_embed_name,
            )
        else:
            merged_in = base_in
            merged_out = base_out

        self._out_tensors[input_embed_name] = merged_in
        self._out_tensors[output_embed_name] = merged_out

    def _merge_tensor(
        self,
        base_tensor: torch.Tensor,
        donor_tensors: List[torch.Tensor],
        method_params: List[torch.Tensor],
        tensor_name: str,
    ) -> torch.Tensor:
        base_f32 = base_tensor.detach().to(torch.float32)
        donors_f32 = [t.detach().to(torch.float32) for t in donor_tensors]

        merged_cpu = self._wtv_grad.merge_tensor(
            base_tensor=base_f32,
            tensors_to_merge=donors_f32,
            method_params=method_params,
            tensor_name=tensor_name,
        )
        logger.info(f"[Debug] Out tensor {tensor_name} has nan: {torch.isnan(merged_cpu).any().item()}")
        return merged_cpu.to(self.device)

    def _merge_cluster(self, spec: Dict) -> None:
        """
        Merge all members of a cluster (across base+donor models) into the base leader layer via ONE WTV call.
        spec = { "cluster_idx": int, "leader": int, "donor_refs": List[(model_name, layer_idx)] }
        """
        leader = int(spec["leader"])
        weight_names = self._get_matches_weight_names(str(leader), match_layer=True)
        w = torch.sigmoid(self.merging_weights[f"cluster_{spec['cluster_idx']}_weights"]).to("cpu")
        w_list = [w[i] for i in range(w.numel())]  # same param vector used for every tensor in this cluster

        for weight_name in weight_names:
            # Base leader tensor
            base_t = self.base_model_cache.get_tensor(weight_name)

            donor_ts: List[torch.Tensor] = []
            # 1) Base model additional layers in cluster
            for (model_name, li) in spec["donor_refs"]:
                if model_name == self.base_model:
                    ow_name = weight_name.replace(f"layers.{leader}.", f"layers.{li}.")
                    donor_ts.append(self.base_model_cache.get_tensor(ow_name))
                else:
                    # donor model tensor of the SAME submodule but at donor layer 'li'
                    ow_name = weight_name.replace(f"layers.{leader}.", f"layers.{li}.")
                    donor_ts.append(self.merging_model_caches[model_name].get_tensor(ow_name))

            merged = self._merge_tensor(
                base_tensor=base_t,
                donor_tensors=donor_ts,
                method_params=w_list,
                tensor_name=weight_name,
            )
            self._out_tensors[weight_name] = merged

    def _merge_postweights(self) -> None:
        post = self._get_matches_weight_names(r"model\.norm\.weight")
        if not post:
            return
        assert len(post) == 1, f"Expected one post-norm weight, found: {post}"
        name = post[0]

        # Use embedding donor weights for post-norm as a simple, stable choice
        if "embed_weights" in self.merging_weights and len(self.merging_models) > 0:
            ew = torch.sigmoid(self.merging_weights["embed_weights"]).to("cpu")
            ew_list = [ew[i] for i in range(ew.numel())]
            base_t = self.base_model_cache.get_tensor(name)
            donor_ts = [self.merging_model_caches[m].get_tensor(name) for m in self.merging_models]
            self._out_tensors[name] = self._merge_tensor(
                base_tensor=base_t,
                donor_tensors=donor_ts,
                method_params=ew_list,
                tensor_name=name,
            )
        else:
            self._out_tensors[name] = self.base_model_cache.get_tensor(name)

    def _update_output_config(self, num_layers_kept=None) -> None:
        if num_layers_kept is not None:
            self._output_config.update({"num_hidden_layers": int(num_layers_kept)})
        else:
            self._output_config.update({"num_hidden_layers": int(self.num_total_layers)})
        self._output_config.update({"vocab_size": int(len(self._aligned_tokenizer.get_vocab()))})

    def _correct_out_tensor_names(self, remaining_indices: List[int]):
        logger.info(f"[ContinuousMerger] Remaining leader indices before renumbering: {remaining_indices}")
        # Map old indices -> new compact indices [0..K-1] in the given order
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_indices)}
        layer_pat = re.compile(r"model\.layers\.(\d+)\.")
        new_tensors = {}
        for name, tensor in self._out_tensors.items():
            m = layer_pat.search(name)
            if m:
                old_idx = int(m.group(1))
                if old_idx in index_map:
                    new_name = name.replace(f"layers.{old_idx}.", f"layers.{index_map[old_idx]}.")
                    new_tensors[new_name] = tensor
            else:
                new_tensors[name] = tensor
        self._out_tensors = new_tensors
        self._output_config.num_hidden_layers = len(remaining_indices)
        logger.info(f"[ContinuousMerger] Final number of layers: {self._output_config.num_hidden_layers}")
        logger.info(f"[ContinuousMerger] Layers retained (leaders old idx): {set(remaining_indices)}")

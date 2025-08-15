# pip install "transformers>=4.41" "datasets>=2.19" torch

import math
import random
from contextlib import contextmanager
from typing import Dict, List, Callable, Optional

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from utils import logger

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class EntropyEvaluator:
    def __init__(
        self,
        *,
        use_vllm: bool = False,                 # must be False when training (vLLM has no autograd path)
        device: str = "cuda",
        dtype: str = "float16",
        mem_log_every: int = 20,                # caller can use this cadence; probes here log every time
        peak_log_every: int = 5,                # caller uses for periodic peak wraps; we wrap forward here
        tag: str = "ev"
    ):
        self.use_vllm = use_vllm
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dtype = DTYPE_MAP.get(dtype, torch.float16)
        self.mem_log_every = max(1, int(mem_log_every))
        self.peak_log_every = max(1, int(peak_log_every))
        self.tag = tag

        self.model: Optional[AutoModelForCausalLM] = None
        self._arch_sig = None  # to avoid re-building model unnecessarily

    # ------------------ GPU memory helpers ------------------
    @staticmethod
    def _bytes(n: int) -> str:
        s = float(n)
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if s < 1024.0:
                return f"{s:.2f}{u}"
            s /= 1024.0
        return f"{s:.2f}PB"

    def _gpu_mem_report(self, tag: str = ""):
        if self.device != "cuda" or not torch.cuda.is_available():
            logger.info(f"[{self.tag}][gpu] {tag}: (no CUDA)")
            return
        dev = torch.cuda.current_device()
        logger.info(
            f"[{self.tag}][gpu] {tag} "
            f"alloc={self._bytes(torch.cuda.memory_allocated(dev))} "
            f"resv={self._bytes(torch.cuda.memory_reserved(dev))} "
            f"max_alloc={self._bytes(torch.cuda.max_memory_allocated(dev))}"
        )

    @contextmanager
    def _gpu_peak_scope(self, tag: str = ""):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            start = None
        try:
            yield
        finally:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                peak_delta = max(0, peak - (start or 0))
                logger.info(f"[{self.tag}][gpu-peak] {tag} Δpeak={self._bytes(peak_delta)}")

    # ------------------ model / overrides ------------------
    def _arch_signature(self, arch_info):
        d = arch_info.to_dict() if hasattr(arch_info, "to_dict") else dict(arch_info)
        return (
            d.get("model_type"),
            d.get("hidden_size"),
            d.get("num_hidden_layers"),
            d.get("vocab_size"),
        )

    def _ensure_model(self, arch_info, tokenizer):
        assert not self.use_vllm, "vLLM does not support autograd; set use_vllm=False for training."
        sig = self._arch_signature(arch_info)
        if self.model is not None and sig == self._arch_sig:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return

        self._gpu_mem_report("before build")
        self.model = AutoModelForCausalLM.from_config(
            arch_info, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._arch_sig = sig
        self._gpu_mem_report("after build")

    def _prepare_param_overrides(self, out_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map your merged weights -> model param names; move to device/dtype (keeps graph)."""
        overrides = {}
        name_to_param = dict(self.model.named_parameters())
        for name, v in out_tensors.items():
            if "rotary_emb.inv_freq" in name:
                continue
            target = name
            if target not in name_to_param and name == "lm_head.weight":
                if "model.embed_tokens.weight" in name_to_param:
                    target = "model.embed_tokens.weight"
                else:
                    continue
            p = name_to_param.get(target)
            if p is None:
                continue
            overrides[target] = v.to(device=p.device, dtype=p.dtype)
        return overrides

    # ------------------ dataset loaders (unchanged) ------------------
    def _resolve_split(self, dataset_id: str, *, config: str | None = None, prefer: List[str] | None = None) -> str:
        prefer = prefer or ["validation", "dev", "test", "train"]
        try:
            avail = list(get_dataset_split_names(dataset_id, config_name=config))
        except TypeError:
            avail = list(get_dataset_split_names(dataset_id, config))
        except Exception:
            avail = []
        if not avail:
            for s in prefer:
                try:
                    _ = load_dataset(dataset_id, name=config, split=f"{s}[:1]")
                    return s
                except Exception:
                    pass
            raise ValueError(f"No splits available for {dataset_id!r} (config={config!r}).")
        for s in prefer:
            if s in avail:
                return s
        return avail[0]

    def _load_piqa(self, limit=100, split="validation", seed=42):
        ds_split = split or self._resolve_split("piqa", prefer=["validation", "test", "train"])
        ds = load_dataset("piqa", split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
        return [{
            "context": ex["goal"],
            "choices": [" " + ex["sol1"], " " + ex["sol2"]],
            "answer_idx": int(ex["label"]),
        } for ex in ds]

    def _load_csqa(self, limit=100, split="validation", seed=42):
        ds_id = "commonsense_qa"
        ds_split = split or self._resolve_split(ds_id, prefer=["validation", "test", "train"])
        ds = load_dataset(ds_id, split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
        out = []
        for ex in ds:
            labels = list(ex["choices"]["label"])
            texts  = list(ex["choices"]["text"])
            out.append({
                "context": ex["question"],
                "choices": [" " + t for t in texts],
                "answer_idx": labels.index(ex["answerKey"]),
            })
        return out

    def _load_mmlu(self, limit=100, split="validation", subjects="auto", seed=42):
        rng = random.Random(seed)
        ds_id = "lukaemon/mmlu"
        if subjects == "auto":
            try:
                subjects = get_dataset_config_names(ds_id)
            except Exception:
                subjects = [
                    "philosophy","high_school_biology","high_school_chemistry",
                    "high_school_physics","college_mathematics",
                    "professional_law","computer_security","econometrics"
                ]
        elif isinstance(subjects, str):
            subjects = [subjects]
        rng.shuffle(subjects)
        per_subj = max(1, (limit + len(subjects) - 1) // len(subjects))
        lut = {"A":0,"B":1,"C":2,"D":3}
        collected = []
        for subj in subjects:
            if len(collected) >= limit:
                break
            use_split = self._resolve_split(ds_id, config=subj, prefer=[split, "validation", "test", "train"])
            ds = load_dataset(ds_id, name=subj, split=f"{use_split}[:{per_subj}]")
            for ex in ds:
                if len(collected) >= limit:
                    break
                q = ex.get("input") or ex.get("question") or ex.get("prompt") or ex.get("query")
                if all(k in ex for k in ("A","B","C","D")):
                    choices = [" " + ex["A"], " " + ex["B"], " " + ex["C"], " " + ex["D"]]
                else:
                    raw = ex.get("choices") or ex.get("options")
                    if isinstance(raw, dict):
                        raw = raw.get("text") or raw.get("options") or list(raw.values())
                    choices = [" " + str(c) for c in raw]
                ans = ex.get("target") or ex.get("answer") or ex.get("label")
                if isinstance(ans, str) and ans.strip() in lut:
                    answer_idx = lut[ans.strip()]
                else:
                    answer_idx = int(ans)
                collected.append({"context": q, "choices": choices, "answer_idx": answer_idx})
        rng.shuffle(collected)
        return collected[:limit]

    def _load_wsc(self, limit=100, split="validation", seed=42):
        ds_split = split or self._resolve_split("super_glue", config="wsc", prefer=["validation","train"])
        ds = load_dataset("super_glue", "wsc", split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
        out = []
        for ex in ds:
            ctx = (
                f"Sentence: {ex['text']}\n"
                f"Does '{ex['span2_text']}' refer to '{ex['span1_text']}'?\n"
                "Answer:"
            )
            out.append({
                "context": ctx,
                "choices": [" Yes", " No"],
                "answer_idx": int(ex["label"]),
            })
        return out

    def _get_loader(self, task: str) -> Callable[..., List[Dict]]:
        task = task.lower()
        if task == "piqa": return self._load_piqa
        if task == "csqa": return self._load_csqa
        if task == "mmlu": return self._load_mmlu
        if task == "wsc":  return self._load_wsc
        raise ValueError(f"Unknown task '{task}'")

    # ------------------ core loss for one batch (with probes) ------------------
    def _entropy_batch_loss(self, model_callable, tokenizer, samples, *, max_len: int, length_norm: bool, base):
        device = self.device

        # PROBE: start
        logger.info(f"[{self.tag}][loss] start: batch_size={len(samples)} max_len={max_len} len_norm={length_norm}")
        self._gpu_mem_report("entropy_batch_loss:start")

        # (sample_idx, choice_idx, ctx_ids, ch_ids, ch_len)
        pairs = []
        for s_idx, s in enumerate(samples):
            ctx_ids = tokenizer(s["context"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            for c_idx, ch in enumerate(s["choices"]):
                ch_ids = tokenizer(ch, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                pairs.append((s_idx, c_idx, ctx_ids, ch_ids, int(ch_ids.numel())))
        if not pairs:
            zero = torch.zeros([], device=device, dtype=self.dtype)
            logger.info(f"[{self.tag}][loss] empty batch -> 0")
            return zero

        seqs, kept_choice_lens, owners = [], [], []
        for (s_idx, c_idx, ctx_ids, ch_ids, ch_len) in pairs:
            ctx_len = int(ctx_ids.numel()); total = ctx_len + ch_len
            if total <= max_len:
                input_ids = torch.cat([ctx_ids, ch_ids], dim=0); kept_ch_len = ch_len
            else:
                keep_ctx = max(0, max_len - ch_len)
                kept_ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx > 0 else ctx_ids.new_empty(0, dtype=ctx_ids.dtype)
                if ch_len > max_len:
                    kept_ch_ids = ch_ids[-max_len:]; kept_ch_len = int(kept_ch_ids.numel()); kept_ctx_ids = ctx_ids.new_empty(0, dtype=ctx_ids.dtype)
                else:
                    kept_ch_ids = ch_ids; kept_ch_len = ch_len
                input_ids = torch.cat([kept_ctx_ids, kept_ch_ids], dim=0)
            seqs.append(input_ids); kept_choice_lens.append(kept_ch_len); owners.append((s_idx, c_idx))

        pad_id = tokenizer.pad_token_id
        maxL = max(int(x.numel()) for x in seqs)
        batch_ids = torch.full((len(seqs), maxL), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros_like(batch_ids)
        lengths = []
        tot_tokens = 0
        for j, ids in enumerate(seqs):
            L = int(ids.numel())
            batch_ids[j, :L] = ids.to(device)
            attn_mask[j, :L] = 1
            lengths.append(L)
            tot_tokens += L

        # PROBE: after allocating batch tensors
        logger.info(f"[{self.tag}][loss] tensors alloc: B={len(seqs)} maxL={maxL} tot_tokens={tot_tokens}")
        self._gpu_mem_report("entropy_batch_loss:after_batch_tensors")

        # Forward (track peak)
        with self._gpu_peak_scope("forward(batch)"):
            out = model_callable(input_ids=batch_ids, attention_mask=attn_mask)  # -> ModelOutput

        # PROBE: after forward
        self._gpu_mem_report("entropy_batch_loss:after_forward")

        logits = out.logits[:, :-1, :]
        targets = batch_ids[:, 1:]

        num_samples = len(samples)
        per_sample_scores = [None] * num_samples
        choice_counts = [len(s["choices"]) for s in samples]

        for j, (s_idx, c_idx) in enumerate(owners):
            T = lengths[j] - 1
            kept_c = kept_choice_lens[j]
            if kept_c <= 0 or T <= 0 or kept_c > T:
                score = logits.new_tensor([-1e30])
            else:
                start = T - kept_c
                lp = F.log_softmax(logits[j, start:start+kept_c, :], dim=-1)
                tgt = targets[j, start:start+kept_c]
                tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                score = tok_lp.mean(0, keepdim=True) if length_norm else tok_lp.sum(0, keepdim=True)
            if per_sample_scores[s_idx] is None:
                per_sample_scores[s_idx] = torch.empty(choice_counts[s_idx], dtype=torch.float32, device=device)
            per_sample_scores[s_idx][c_idx] = score.squeeze(0).to(torch.float32)

        entropies = []
        for scores in per_sample_scores:
            probs = torch.softmax(scores, dim=-1)
            ent = -(probs * probs.clamp_min(1e-20).log()).sum()
            if base == 2:
                ent = ent / math.log(2)
            entropies.append(ent)
        loss = torch.stack(entropies).sum()

        # PROBE: after loss build
        logger.info(f"[{self.tag}][loss] built: scalar={float(loss.detach().cpu()):.6f} (nats if base=e)")
        self._gpu_mem_report("entropy_batch_loss:after_loss")

        # Drop local refs before returning (graph stays via `loss`)
        try:
            del seqs, kept_choice_lens, owners, batch_ids, attn_mask, logits, targets, per_sample_scores, entropies
        except Exception:
            pass

        return loss

    # ------------------ single-batch API (no backward) with probes ------------------
    def batch_loss(
        self,
        *,
        out_tensors: Dict[str, torch.Tensor],
        arch_info,
        tokenizer,
        samples: List[Dict],
        max_len: int = 2048,
        length_norm: bool = False,
        base=math.e,
    ) -> torch.Tensor:
        """
        Compute and return the loss for exactly ONE batch (list of samples).
        - No internal batching
        - No optimizer logic
        - No .backward()
        Caller handles batching, backward, step(), and re-merge.
        """
        # PROBE: start of batch_loss
        logger.info(f"[{self.tag}][batch_loss] start: batch_size={len(samples)}")
        self._ensure_model(arch_info, tokenizer)

        # Map merged tensors to model params (keeps graph + devices aligned)
        overrides = self._prepare_param_overrides(out_tensors)

        # PROBE: overrides stats
        n_over = len(overrides)
        n_grad = sum(int(t.requires_grad) for t in overrides.values())
        bytes_total = 0
        on_cuda = 0
        for t in overrides.values():
            bytes_total += t.element_size() * t.numel()
            if t.is_cuda:
                on_cuda += 1
        logger.info(f"[{self.tag}][batch_loss] overrides: {n_over} "
                    f"(gradful={n_grad}) size≈{self._bytes(bytes_total)} on_cuda={on_cuda}")
        self._gpu_mem_report("batch_loss:after_overrides")

        def _model_with_overrides(**kwargs):
            return functional_call(self.model, overrides, args=(), kwargs={**kwargs, "use_cache": False})

        loss = self._entropy_batch_loss(
            model_callable=_model_with_overrides,
            tokenizer=tokenizer,
            samples=samples,
            max_len=max_len,
            length_norm=length_norm,
            base=base,
        )

        # PROBE: end of batch_loss
        logger.info(f"[{self.tag}][batch_loss] done: requires_grad={loss.requires_grad} grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}")
        self._gpu_mem_report("batch_loss:end")

        # Drop local refs (loss keeps graph)
        try:
            del overrides
        except Exception:
            pass

        return loss

# evaluation/entropy_trainer.py

import math
from typing import Dict, List, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

from utils import logger

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class EntropyTrainer:
    """
    Differentiable multiple-choice entropy loss with internal dataset loading.
      - Call `num_batches(task, ...)`
      - Call `batch_loss_by_index(..., task=..., batch_idx=..., batch_size=...)`
    Uses `num_logits_to_keep` to cap the logits window to the needed choice length,
    greatly reducing peak GPU memory without changing the loss.
    """

    def __init__(self, *, device: str = "cuda", dtype: str = "float16"):
        self.device = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.dtype = DTYPE_MAP.get(dtype, torch.float16)

        self.model: Optional[AutoModelForCausalLM] = None
        self._arch_sig: Optional[Tuple] = None
        self._task_cache: dict[Tuple, List[Dict]] = {}  # (task, split, limit, seed, subjects) -> samples

    # ---------- model / overrides ----------
    def _arch_signature(self, arch_info):
        d = arch_info.to_dict() if hasattr(arch_info, "to_dict") else dict(arch_info)
        return (
            d.get("model_type"),
            d.get("hidden_size"),
            d.get("num_hidden_layers"),
            d.get("vocab_size"),
        )

    def _ensure_model(self, arch_info, tokenizer):
        sig = self._arch_signature(arch_info)
        if self.model is not None and sig == self._arch_sig:
            # ensure pad token for batching
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return

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

    def _prepare_param_overrides(self, out_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map merged tensors -> model param names; move to device/dtype (keeps graph)."""
        overrides = {}
        name_to_param = dict(self.model.named_parameters())
        for name, v in out_tensors.items():
            if "rotary_emb.inv_freq" in name:
                continue
            target = name
            # tie lm_head to embed_tokens if needed
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

    # ---------- dataset helpers ----------
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
            raise ValueError(f"No splits for {dataset_id!r} (config={config!r}).")
        for s in prefer:
            if s in avail:
                return s
        return avail[0]

    def _load_piqa(self, limit=100, split="validation", seed=42):
        ds_split = split or self._resolve_split("piqa", prefer=["validation", "test", "train"])
        ds = load_dataset("piqa", split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
        return [{"context": ex["goal"], "choices": [" " + ex["sol1"], " " + ex["sol2"]], "answer_idx": int(ex["label"])}
                for ex in ds]

    def _load_csqa(self, limit=100, split="validation", seed=42):
        ds_id = "commonsense_qa"
        ds_split = split or self._resolve_split(ds_id, prefer=["validation", "test", "train"])
        ds = load_dataset(ds_id, split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
        out = []
        for ex in ds:
            labels = list(ex["choices"]["label"])
            texts  = list(ex["choices"]["text"])
            out.append({"context": ex["question"], "choices": [" " + t for t in texts],
                        "answer_idx": labels.index(ex["answerKey"])})
        return out

    def _load_mmlu(self, limit=100, split="validation", subjects="auto", seed=42):
        import random
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
                answer_idx = lut[ans.strip()] if isinstance(ans, str) and ans.strip() in lut else int(ans)
                collected.append({"context": q, "choices": choices, "answer_idx": answer_idx})
        rng.shuffle(collected)
        return collected[:limit]

    def _load_wsc(self, limit=100, split="validation", seed=42):
        ds_split = self._resolve_split("super_glue", config="wsc", prefer=["validation","train"])
        ds = load_dataset("super_glue", "wsc", split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
        out = []
        for ex in ds:
            ctx = (
                f"Sentence: {ex['text']}\n"
                f"Does '{ex['span2_text']}' refer to '{ex['span1_text']}'?\n"
                "Answer:"
            )
            out.append({"context": ctx, "choices": [" Yes", " No"], "answer_idx": int(ex["label"])})
        return out

    def _get_loader(self, task: str) -> Callable[..., List[Dict]]:
        t = task.lower()
        if t == "piqa": return self._load_piqa
        if t == "commonsense_qa": return self._load_csqa
        if t == "mmlu": return self._load_mmlu
        if t == "wsc":  return self._load_wsc
        raise ValueError(f"Unknown task '{task}'")

    def _get_or_make_samples(
        self, task: str, *, limit: int = 100, split: str = "validation", seed: int = 42, subjects="auto"
    ) -> List[Dict]:
        key = (task.lower(), split, int(limit), int(seed), str(subjects))
        if key in self._task_cache:
            return self._task_cache[key]
        loader = self._get_loader(task)
        if task.lower() == "mmlu":
            data = loader(limit=limit, split=split, subjects=subjects, seed=seed)
        else:
            data = loader(limit=limit, split=split, seed=seed)
        self._task_cache[key] = data
        return data

    def num_batches(
        self, task: str, *, batch_size: int, limit: int = 100, split: str = "validation", seed: int = 42, subjects="auto"
    ) -> int:
        n = len(self._get_or_make_samples(task, limit=limit, split=split, seed=seed, subjects=subjects))
        return (n + batch_size - 1) // batch_size

    # ---------- core loss for one batch (memory-friendly) ----------
    def _entropy_batch_loss(self, model_callable, tokenizer, samples, *, max_len: int, length_norm: bool, base):
        """
        Builds a single multiple-choice batch and computes the entropy loss.
        Uses `num_logits_to_keep=K` where K = max(choice_len)+1 to reduce memory.
        """
        device = self.device

        # (sample_idx, choice_idx, ctx_ids, ch_ids, ch_len)
        pairs = []
        for s_idx, s in enumerate(samples):
            ctx_ids = tokenizer(s["context"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            for c_idx, ch in enumerate(s["choices"]):
                ch_ids = tokenizer(ch, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                pairs.append((s_idx, c_idx, ctx_ids, ch_ids, int(ch_ids.numel())))

        if not pairs:
            return torch.zeros([], device=device, dtype=self.dtype)

        seqs, kept_choice_lens, owners = [], [], []
        for (s_idx, c_idx, ctx_ids, ch_ids, ch_len) in pairs:
            ctx_len = int(ctx_ids.numel()); total = ctx_len + ch_len
            if total <= max_len:
                input_ids = torch.cat([ctx_ids, ch_ids], dim=0); kept_ch_len = ch_len
            else:
                keep_ctx = max(0, max_len - ch_len)
                kept_ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx > 0 else ctx_ids.new_empty(0, dtype=ctx_ids.dtype)
                if ch_len > max_len:
                    kept_ch_ids = ch_ids[-max_len:]; kept_ch_len = int(kept_ch_ids.numel())
                    kept_ctx_ids = ctx_ids.new_empty(0, dtype=ctx_ids.dtype)
                else:
                    kept_ch_ids = ch_ids; kept_ch_len = ch_len
                input_ids = torch.cat([kept_ctx_ids, kept_ch_ids], dim=0)
            seqs.append(input_ids); kept_choice_lens.append(kept_ch_len); owners.append((s_idx, c_idx))

        pad_id = tokenizer.pad_token_id
        maxL = max(int(x.numel()) for x in seqs)
        batch_ids = torch.full((len(seqs), maxL), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros_like(batch_ids)
        lengths = []
        for j, ids in enumerate(seqs):
            L = int(ids.numel())
            batch_ids[j, :L] = ids.to(device)
            attn_mask[j, :L] = 1
            lengths.append(L)

        # Memory saver: only keep logits for the last K steps.
        # K must cover the scored choice tokens (+1 for AR shift).
        K = max(1, max(kept_choice_lens) + 1)

        # Some models don't accept num_logits_to_keep; fall back if needed.
        try:
            out = model_callable(input_ids=batch_ids, attention_mask=attn_mask, num_logits_to_keep=K)
        except TypeError:
            out = model_callable(input_ids=batch_ids, attention_mask=attn_mask)

        # Align logits/targets with standard next-token shift, but only in the kept window
        raw_logits = out.logits  # [B, K, V] if num_logits_to_keep worked, else [B, T, V]
        if raw_logits.size(1) > 0 and raw_logits.size(1) <= K:  # trimmed case (or small T)
            logits = raw_logits[:, :-1, :]                       # [B, K-1, V]
            targets = batch_ids[:, -logits.size(1):]             # last K-1 tokens as labels
        else:
            # Fallback to full sequence behavior
            logits = raw_logits[:, :-1, :]
            targets = batch_ids[:, 1:]

        num_samples = len(samples)
        per_sample_scores = [None] * num_samples
        choice_counts = [len(s["choices"]) for s in samples]

        T_kept = logits.size(1)
        for j, (s_idx, c_idx) in enumerate(owners):
            kept_c = kept_choice_lens[j]
            if kept_c <= 0 or T_kept <= 0 or kept_c > T_kept:
                score = logits.new_tensor([-1e30])
            else:
                row_logits  = logits[j, -kept_c:, :]      # last kept_c steps
                row_targets = targets[j, -kept_c:]
                lp = F.log_softmax(row_logits, dim=-1)
                tok_lp = lp.gather(-1, row_targets.unsqueeze(-1)).squeeze(-1)
                score = tok_lp.mean(0, keepdim=True) if length_norm else tok_lp.sum(0, keepdim=True)
            if per_sample_scores[s_idx] is None:
                per_sample_scores[s_idx] = torch.empty(choice_counts[s_idx], dtype=torch.float32, device=device)
            per_sample_scores[s_idx][c_idx] = score.squeeze(0).to(torch.float32)

        entropies = []
        for scores in per_sample_scores:
            C = scores.numel()
            probs = torch.softmax(scores, dim=-1)
            ent = -(probs * probs.clamp_min(1e-20).log()).sum()
            # if base == 2:
            #     ent = ent / math.log(2)

            if C > 1:
                ent = ent / math.log(C)
            entropies.append(ent)
        loss = torch.stack(entropies).mean()

        # Drop local refs before returning (graph is kept via `loss`)
        try:
            del seqs, kept_choice_lens, owners, batch_ids, attn_mask, logits, targets, per_sample_scores, entropies
        except Exception:
            pass

        return loss

    # ---------- public API with internal loading ----------
    def batch_loss_by_index(
        self,
        *,
        out_tensors: Dict[str, torch.Tensor],
        arch_info,
        tokenizer,
        task: str,
        batch_idx: int,
        batch_size: int,
        max_len: int = 2048,
        length_norm: bool = True,
        base=math.e,
        limit: int = 100,
        split: str = "validation",
        seed: int = 42,
        subjects="auto",  # only used for MMLU
    ) -> torch.Tensor:
        """
        Load/cache the dataset internally, slice the requested batch, and compute loss.
        Returns a scalar tensor with grad.
        """
        self._ensure_model(arch_info, tokenizer)
        overrides = self._prepare_param_overrides(out_tensors)

        data = self._get_or_make_samples(task, limit=limit, split=split, seed=seed, subjects=subjects)
        start = batch_idx * batch_size
        end = min(start + batch_size, len(data))
        if start >= len(data):
            return torch.zeros([], device=self.device, dtype=self.dtype)

        batch = data[start:end]

        def _model_with_overrides(**kwargs):
            # functional_call keeps autograd path from overrides -> loss
            return functional_call(self.model, overrides, args=(), kwargs={**kwargs, "use_cache": False})

        loss = self._entropy_batch_loss(
            model_callable=_model_with_overrides,
            tokenizer=tokenizer,
            samples=batch,
            max_len=max_len,
            length_norm=length_norm,
            base=base,
        )

        try:
            del overrides
        except Exception:
            pass

        return loss

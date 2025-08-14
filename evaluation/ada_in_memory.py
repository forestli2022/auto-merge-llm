# python 3.10+
# pip install "transformers>=4.41" "datasets>=2.19" torch accelerate

from contextlib import nullcontext
import os, math, gc
import random
from typing import Dict, List
import torch
import torch.nn.functional as F
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
) 
from torch.func import functional_call

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

# -------------------- 1) Build frozen model from config + out_tensors --------------------
def _build_frozen_model(arch_info, out_tensors: Dict[str, torch.Tensor]):
    model = AutoModelForCausalLM.from_config(
        arch_info, trust_remote_code=True, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _prepare_param_overrides(model, out_tensors: Dict[str, torch.Tensor]):
    """Coerce your differentiable out_tensors to the model's param names/device/dtype."""
    overrides = {}
    name_to_param = dict(model.named_parameters())
    for name, v in out_tensors.items():
        if "rotary_emb.inv_freq" in name:
            continue  # buffer; not needed for params
        # handle possible tie/missing head name gracefully
        if name not in name_to_param and name == "lm_head.weight" and "lm_head.weight" not in name_to_param:
            # some archs tie lm_head to embed_tokens; if your arch actually lacks lm_head, skip this
            if "model.embed_tokens.weight" in name_to_param:
                target = "model.embed_tokens.weight"
            else:
                continue
        else:
            target = name

        p = name_to_param.get(target, None)
        if p is None:
            continue
        overrides[target] = v.to(device=p.device, dtype=p.dtype)
    return overrides

# -------------------- 2) Dataset mappers -> {context, choices, answer_idx} --------------------
def peek_keys(dataset_id: str, *, config: str | None = None, split: str | None = None, n: int = 2):
    """
    Print features, column names, and example keys for a dataset/config/split.
    """
    use_split = split or _resolve_split(dataset_id, config=config)
    ds = load_dataset(dataset_id, name=config, split=f"{use_split}[:{n}]")
    print(f"\n=== {dataset_id} | config={config} | split={use_split} ===")
    print("features:", ds.features)
    print("columns:", ds.column_names)
    for i, ex in enumerate(ds):
        print(f"\n-- example[{i}] keys:", list(ex.keys()))
        # If choices look nested, show their structure briefly
        if "choices" in ex:
            print("choices type:", type(ex["choices"]))
            if isinstance(ex["choices"], dict):
                print("choices dict keys:", list(ex["choices"].keys()))
        print("example snippet:", {k: (str(v)[:120] + ("..." if len(str(v)) > 120 else "")) for k, v in ex.items()})

def _resolve_split(dataset_id: str, *, config: str | None = None, prefer: list[str] = None):
    if prefer is None:
        prefer = ["validation", "dev", "test", "train"]
    try:
        avail = list(get_dataset_split_names(dataset_id, config_name=config))
    except TypeError:
        # older versions sometimes accept positional config
        avail = list(get_dataset_split_names(dataset_id, config))
    except Exception:
        avail = []
    if not avail:
        # last resort: probe quickly
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

# ---------- PIQA ----------
def _load_piqa(limit: int = 100, split: str | None = "validation", seed: int = 42):
    ds_split = split or _resolve_split("piqa", prefer=["validation", "test", "train"])
    ds = load_dataset("piqa", split=f"{ds_split}[:{limit}]").shuffle(seed=seed)
    return [{
        "context": ex["goal"],
        "choices": [" " + ex["sol1"], " " + ex["sol2"]],
        "answer_idx": int(ex["label"]),  # label is '0'/'1' strings in your dump
    } for ex in ds]

# ---------- CommonsenseQA ----------
def _load_csqa(limit: int = 100, split: str | None = "validation", seed: int = 42):
    dataset_id = "commonsense_qa"
    ds_split = split or _resolve_split(dataset_id, prefer=["validation", "test", "train"])
    ds = load_dataset(dataset_id, split=f"{ds_split}[:{limit}]").shuffle(seed=seed)

    out = []
    for ex in ds:
        # choices is a dict with 'label' and 'text' lists
        labels = list(ex["choices"]["label"])   # ['A','B','C','D','E']
        texts  = list(ex["choices"]["text"])
        out.append({
            "context": ex["question"],
            "choices": [" " + t for t in texts],
            "answer_idx": labels.index(ex["answerKey"]),
        })
    return out

# ---------- MMLU (lukaemon/mmlu) ----------
def _load_mmlu(
    limit: int = 100,
    split: str = "validation",            # your dump showed validation
    subjects: str | list[str] = "auto",
    seed: int = 42,
):
    rng = random.Random(seed)
    dataset_id = "lukaemon/mmlu"

    # discover subjects if needed
    if subjects == "auto":
        try:
            subjects = get_dataset_config_names(dataset_id)
        except Exception:
            subjects = [
                "philosophy", "high_school_biology", "high_school_chemistry",
                "high_school_physics", "college_mathematics",
                "professional_law", "computer_security", "econometrics",
            ]
    elif isinstance(subjects, str):
        subjects = [subjects]

    rng.shuffle(subjects)
    per_subj = max(1, (limit + len(subjects) - 1) // len(subjects))

    lut = {"A": 0, "B": 1, "C": 2, "D": 3}
    collected = []

    for subj in subjects:
        if len(collected) >= limit:
            break
        use_split = _resolve_split(dataset_id, config=subj, prefer=[split, "validation", "test", "train"])
        ds = load_dataset(dataset_id, name=subj, split=f"{use_split}[:{per_subj}]")
        for ex in ds:
            if len(collected) >= limit:
                break

            # In your dump: input + fields A,B,C,D + target (letter)
            q = ex.get("input") or ex.get("question") or ex.get("prompt") or ex.get("query")
            if q is None:
                raise KeyError(f"MMLU[{subj}]: cannot find question/input in keys={list(ex.keys())}")

            if all(k in ex for k in ("A", "B", "C", "D")):
                choices = [" " + ex["A"], " " + ex["B"], " " + ex["C"], " " + ex["D"]]
            else:
                raw = ex.get("choices") or ex.get("options")
                if isinstance(raw, dict):
                    raw = raw.get("text") or raw.get("options") or list(raw.values())
                if not isinstance(raw, (list, tuple)):
                    raise KeyError(f"MMLU[{subj}]: cannot find choices in keys={list(ex.keys())}")
                choices = [" " + str(c) for c in raw]

            ans = ex.get("target") or ex.get("answer") or ex.get("label")
            if ans is None:
                raise KeyError(f"MMLU[{subj}]: missing target/answer/label")

            if isinstance(ans, str):
                a = ans.strip()
                if a in lut:
                    answer_idx = lut[a]
                else:
                    # numeric string or exact choice text
                    try:
                        answer_idx = int(a)
                    except Exception:
                        try:
                            answer_idx = [c.strip() for c in [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]].index(a)
                        except Exception:
                            raise ValueError(f"MMLU[{subj}]: cannot resolve answer index from '{ans}'")
            else:
                answer_idx = int(ans)

            collected.append({
                "context": q,
                "choices": choices,
                "answer_idx": answer_idx,
            })

    rng.shuffle(collected)
    return collected[:limit]

# ---------- WSC (SuperGLUE) ----------
def _load_wsc(limit: int = 100, split: str | None = "validation", seed: int = 42):
    ds_split = split or _resolve_split("super_glue", config="wsc", prefer=["validation", "train"])
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
            "answer_idx": int(ex["label"]),  # label is '0'/'1' strings in your dump
        })
    return out

# -------------------- 3) Batched scorer (frozen LM; no grads) --------------------
def _score_task_batched(
    model, tokenizer, samples: List[Dict],
    batch_size: int = 8, max_len: int = 2048, length_norm: bool = False, base=math.e,
    *, param_overrides: Dict[str, torch.Tensor] | None = None,
    with_grad: bool = True,
):
    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    N = len(samples)
    per_sample_scores: List[torch.Tensor] = [None] * N
    per_sample_choice_count = [len(s["choices"]) for s in samples]

    # Build (sample_idx, choice_idx, ctx_ids, ch_ids, ch_len)
    pairs = []
    for s_idx, s in enumerate(samples):
        ctx_ids = tokenizer(s["context"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for c_idx, ch in enumerate(s["choices"]):
            ch_ids = tokenizer(ch, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            pairs.append((s_idx, c_idx, ctx_ids, ch_ids, int(ch_ids.numel())))

    # Mini-batches
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        seqs, kept_choice_lens, owners = [], [], []
        for (s_idx, c_idx, ctx_ids, ch_ids, ch_len) in chunk:
            ctx_len = int(ctx_ids.numel())
            total = ctx_len + ch_len
            if total <= max_len:
                input_ids = torch.cat([ctx_ids, ch_ids], dim=0)
                kept_ch_len = ch_len
            else:
                keep_ctx = max(0, max_len - ch_len)
                kept_ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx > 0 else ctx_ids.new_empty(0, dtype=ctx_ids.dtype)
                if ch_len > max_len:
                    kept_ch_ids = ch_ids[-max_len:]
                    kept_ch_len = int(kept_ch_ids.numel())
                    kept_ctx_ids = ctx_ids.new_empty(0, dtype=ctx_ids.dtype)
                else:
                    kept_ch_ids = ch_ids
                    kept_ch_len = ch_len
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

        ctx = nullcontext() if with_grad else torch.inference_mode()
        with ctx:
            if param_overrides is not None:
                out = functional_call(
                    model,
                    param_overrides,
                    args=(),
                    kwargs={"input_ids": batch_ids, "attention_mask": attn_mask, "use_cache": False},
                )
            else:
                out = model(input_ids=batch_ids, attention_mask=attn_mask, use_cache=False)

        logits = out.logits[:, :-1, :]
        targets = batch_ids[:, 1:]

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
                per_sample_scores[s_idx] = torch.empty(per_sample_choice_count[s_idx], dtype=torch.float32, device=device)
            per_sample_scores[s_idx][c_idx] = score.squeeze(0).to(torch.float32)

    # scores -> probs -> entropy
    results, correct = [], 0
    ent_sum_float = 0.0
    entropies_tensor = []

    for idx, s in enumerate(samples):
        scores = per_sample_scores[idx]
        probs = torch.softmax(scores, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-20)).log()).sum()
        if base == 2:
            ent = ent / math.log(2)
        entropies_tensor.append(ent)

        pred = int(torch.argmax(probs).item())
        ok = int(pred == s["answer_idx"])
        correct += ok
        ent_sum_float += float(ent.detach().cpu())

        results.append({
            "idx": idx,
            "entropy": float(ent.detach().cpu()),
            "pred": pred,
            "answer_idx": int(s["answer_idx"]),
            "correct": bool(ok),
            "probs": probs.detach().cpu().tolist(),
        })

    # tensor sum (for backprop) vs float sum (for logging)
    task_entropy_sum_tensor = torch.stack(entropies_tensor).sum() if with_grad else None
    summary = {
        "count": len(samples),
        "entropy_sum": ent_sum_float,
        "mean_entropy": ent_sum_float / max(1, len(samples)),
        "accuracy": correct / max(1, len(samples)),
    }
    return results, summary, task_entropy_sum_tensor



# -------------------- 4) High-level: load tasks, batch-score, summarize --------------------
def get_entropy(
    out_tensors, arch_info, aligned_tokenizer,
    *, tasks=("piqa","csqa","mmlu","wsc"),
    limit=100, batch_size=8, max_len=2048, length_norm=False, base=math.e,
    with_grad: bool = True,
):
    model = _build_frozen_model(arch_info, out_tensors)
    tok = aligned_tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Build overrides directly from your (differentiable) out_tensors
    param_overrides = _prepare_param_overrides(model, out_tensors)

    loaders = {
        "piqa": _load_piqa,
        "csqa": _load_csqa,
        "mmlu": lambda limit: _load_mmlu(limit=limit, split="validation", subjects="auto"),
        "wsc":  _load_wsc
    }

    by_task = {}
    overall_items = 0
    overall_acc_num = 0.0
    overall_entropy_sum_float = 0.0
    overall_entropy_sum_tensor = None

    for t in tasks:
        samples = loaders[t](limit=limit)
        results, summary, task_entropy_sum_tensor = _score_task_batched(
            model, tok, samples,
            batch_size=batch_size, max_len=max_len, length_norm=length_norm, base=base,
            param_overrides=param_overrides, with_grad=with_grad
        )
        by_task[t] = {"results": results, "summary": summary}

        overall_items += summary["count"]
        overall_acc_num += summary["accuracy"] * summary["count"]
        overall_entropy_sum_float += summary["entropy_sum"]

        if with_grad:
            overall_entropy_sum_tensor = (
                task_entropy_sum_tensor if overall_entropy_sum_tensor is None
                else overall_entropy_sum_tensor + task_entropy_sum_tensor
            )

    overall = {
        "count": overall_items,
        "entropy_sum": overall_entropy_sum_float,
        "mean_entropy": (overall_entropy_sum_float / max(1, overall_items)),
        "accuracy": (overall_acc_num / max(1, overall_items)),
    }

    return {
        "by_task": by_task,
        "overall": overall,
        # Differentiable: sum of entropies over ALL samples in ALL tasks
        "overall_entropy_sum_tensor": overall_entropy_sum_tensor,
    }



# -------------------- 5) Bootstrapping Llama-3.2-1B into (arch_info, tokenizer, out_tensors) --------------------
def bootstrap_llama32_1b(model_id: str, cache_dir: str | None = None, token: str | None = None):
    """
    Loads config + tokenizer from HF (or local path), then builds out_tensors by
    pulling the state_dict of the base model. This is perfect for sanity tests:
    you can swap out_tensors later with your merged weights.
    """
    arch_info = AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir, token=token)

    # Load base weights to CPU in half precision; extract state_dict -> out_tensors
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="cpu",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        token=token,
    )
    # NOTE: keep on CPU to avoid VRAM hit; our scorer moves tensors as needed
    state = {k: v.to("cpu").contiguous() for k, v in base.state_dict().items()}
    del base; gc.collect()
    return arch_info, tokenizer, state

# -------------------- __main__: set up from your cache + run --------------------
if __name__ == "__main__":
    peek_keys("piqa")                               # PIQA (auto-split)
    peek_keys("commonsense_qa")                     # CSQA
    peek_keys("super_glue", config="wsc")           # WSC
    peek_keys("lukaemon/mmlu", config="philosophy") # MMLU (one subject)
    # Choose one:
    MODEL_ID  = os.environ.get("MODEL_ID",  "meta-llama/Llama-3.2-1B")  # or "meta-llama/Llama-3.2-1B-Instruct"
    # Use a **local** snapshot dir if you have it (overrides MODEL_ID):
    MODEL_PATH = "/mnt/ccnas2/tdp/fl1123/.cache/auto-merge-llm-cache/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
    HF_TOKEN   = os.environ.get("HF_TOKEN")    # required for gated repos if pulling from Hub
    CACHE_DIR  = "/mnt/ccnas2/tdp/fl1123/.cache/auto-merge-llm-cache/"

    model_ref = MODEL_PATH if MODEL_PATH else MODEL_ID

    arch_info, tok, out_tensors = bootstrap_llama32_1b(model_ref, cache_dir=CACHE_DIR, token=HF_TOKEN)

    # quick run (100 items each, batch=8). Tweak as needed.
    results = get_entropy(
        out_tensors, arch_info, tok,
        tasks=("piqa","csqa","mmlu","wsc"),
        limit=10, batch_size=4, max_len=2048, length_norm=False, base=math.e
    )

    print("\n=== OVERALL ===")
    print(results["overall"])
    for t in ("piqa","csqa","mmlu","wsc"):
        print(f"{t.upper()} summary:", results["by_task"][t]["summary"])

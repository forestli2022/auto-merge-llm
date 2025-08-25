from __future__ import annotations
import re
import torch
from utils import logger

def safe_stats(t: torch.Tensor) -> dict:
    t32 = t.detach().to(torch.float32)
    has_nan = torch.isnan(t32).any().item()
    has_inf = torch.isinf(t32).any().item()
    if t32.numel() == 0:
        return dict(nan=has_nan, inf=has_inf, min=None, max=None, mean=None, std=None, rms=None)
    return dict(
        nan=has_nan,
        inf=has_inf,
        min=float(t32.min().item()),
        max=float(t32.max().item()),
        mean=float(t32.mean().item()),
        std=float(t32.std().item()),
        rms=float((t32.pow(2).mean().sqrt()).item()),
    )

def dump_cluster_heads(merger, elems: int = 6, max_blocks: int = 50):
    layer_pat = re.compile(r"model\.layers\.(\d+)\.")
    per_layer = {}
    for name in merger.out_tensors.keys():
        m = layer_pat.search(name)
        if m:
            li = int(m.group(1))
            per_layer.setdefault(li, []).append(name)

    for spec in merger.cluster_specs:
        leader = int(spec["leader"])
        names = sorted(per_layer.get(leader, []))
        logger.info(f"[inspect] cluster {spec['cluster_idx']}  leader={leader}  donors={len(spec['donor_refs'])}")
        for i, n in enumerate(names[:max_blocks]):
            t = merger.out_tensors[n]
            st = safe_stats(t)
            head = t.detach().reshape(-1)[:elems].to(torch.float32).cpu().tolist()
            logger.info(
                f"  - {n}  shape={tuple(t.shape)}  "
                f"nan={st['nan']} inf={st['inf']} "
                f"min={st['min']:.4g} max={st['max']:.4g} mean={st['mean']:.4g} std={st['std']:.4g} rms={st['rms']:.4g} "
                f"head={head}"
            )

def probe_cluster_progressive(merger, cluster_idx: int, subname_filter: str | None = None, elems: int = 6):
    spec = merger.cluster_specs[cluster_idx]
    leader = int(spec["leader"])

    layer_pat = re.compile(rf"model\.layers\.{leader}\.")
    weight_names = [n for n in merger.out_tensors.keys() if layer_pat.search(n)]
    weight_names.sort()
    if not weight_names:
        logger.info(f"[probe] cluster {cluster_idx}: no tensors found for leader layer {leader}")
        return
    if subname_filter:
        weight_names = [n for n in weight_names if subname_filter in n] or weight_names

    target_name = weight_names[0]
    logger.info(f"[probe] cluster {cluster_idx}  leader={leader}  tensor={target_name}")

    base_t = merger.base_model_cache.get_tensor(target_name).to(torch.float32)

    donors = []
    for (model_name, li) in spec["donor_refs"]:
        ow_name = target_name.replace(f"layers.{leader}.", f"layers.{li}.")
        src = merger.base_model_cache if model_name == merger.base_model else merger.merging_model_caches[model_name]
        donors.append(src.get_tensor(ow_name).to(torch.float32))

    w = torch.sigmoid(merger.merging_weights[f"cluster_{spec['cluster_idx']}_weights"]).to(dtype=torch.float32)
    logger.info(f"[probe] weights(sigmoid) head={w[:min(8, w.numel())].tolist()}  (#donors={len(donors)})")

    st_base = safe_stats(base_t)
    logger.info(f"[probe] base stats: nan={st_base['nan']} inf={st_base['inf']} "
                f"min={st_base['min']:.4g} max={st_base['max']:.4g} std={st_base['std']:.4g} rms={st_base['rms']:.4g} "
                f"head={base_t.reshape(-1)[:elems].cpu().tolist()}")

    for di, d in enumerate(donors):
        st_d = safe_stats(d)
        logger.info(f"[probe] donor[{di}] {spec['donor_refs'][di]} stats: nan={st_d['nan']} inf={st_d['inf']} "
                    f"min={st_d['min']:.4g} max={st_d['max']:.4g} std={st_d['std']:.4g} rms={st_d['rms']:.4g} "
                    f"head={d.reshape(-1)[:elems].cpu().tolist()}")

    cur = base_t
    for i in range(len(donors)):
        merged_i = merger._wtv_grad.merge_tensor(
            base_tensor=cur,
            tensors_to_merge=[donors[i]],
            method_params=[w[i]],
            tensor_name=target_name,
        ).to(torch.float32)

        st = safe_stats(merged_i)
        logger.info(f"[probe] after donor {i} {spec['donor_refs'][i]} -> "
                    f"nan={st['nan']} inf={st['inf']} min={st['min']:.4g} max={st['max']:.4g} "
                    f"std={st['std']:.4g} rms={st['rms']:.4g} "
                    f"head={merged_i.reshape(-1)[:elems].cpu().tolist()}")
        cur = merged_i

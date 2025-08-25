# utils/probe.py
import torch
from typing import Any, Optional

def _resolve_device(dev: Optional[Any] = None) -> Optional[int]:
    """
    Return a CUDA device index (int) or None if CUDA unavailable.
    Accepts int, str ('cuda' / 'cuda:0'), torch.device, or None.
    """
    if not torch.cuda.is_available():
        return None
    if dev is None:
        return torch.cuda.current_device()
    if isinstance(dev, int):
        return dev
    if isinstance(dev, str):
        d = torch.device(dev)
        if d.type != "cuda":
            return None
        return d.index if d.index is not None else 0
    if isinstance(dev, torch.device):
        if dev.type != "cuda":
            return None
        return dev.index if dev.index is not None else 0
    # Fallback
    return torch.cuda.current_device()

def probe(msg: str, device: Optional[Any] = None, logger=None) -> None:
    """
    Log CUDA allocated/reserved/max for the given device (or current device).
    Never raises — logs a friendly error if probing fails or CUDA is absent.
    """
    try:
        idx = _resolve_device(device)
        if idx is None:
            if logger:
                logger.info(f"[cuda-mem] {msg} | CUDA not available")
            return
        alloc = torch.cuda.memory_allocated(idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated(idx) / (1024**3)
        if logger:
            logger.info(f"[cuda-mem] {msg} | alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB max={max_alloc:.2f}GiB")
    except Exception as e:
        if logger:
            logger.info(f"[cuda-mem] {msg} | probe failed: {e}")

def empty_cache(device: Optional[Any] = None, logger=None) -> None:
    """
    Empty CUDA cache on the resolved device (no-op if CUDA absent).
    """
    try:
        idx = _resolve_device(device)
        if idx is None:
            return
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()
    except Exception as e:
        if logger:
            logger.info(f"[cuda-mem] empty_cache failed: {e}")

"""
LR scheduler factory — shared by all foundry trainers.
"""
from __future__ import annotations

import math


def build_scheduler(optimizer, lr_scheduler: str, warmup_steps: int, total_steps: int):
    """
    Build a PyTorch LambdaLR scheduler from a string name.

    Schedules:
        "constant"  — fixed LR (with optional linear warmup if warmup_steps > 0).
        "cosine"    — cosine decay from LR to 0 after warmup.
        "linear"    — linear decay from LR to 0 after warmup.

    Args:
        optimizer:    The optimizer whose LR will be scheduled.
        lr_scheduler: One of "constant", "cosine", "linear".
        warmup_steps: Number of optimizer steps for linear LR ramp-up.
        total_steps:  Total optimizer steps (used for decay schedules).
                      If 0, falls back to constant LR.

    Returns:
        A ``torch.optim.lr_scheduler.LambdaLR`` instance, or ``None`` for
        pure constant LR (no warmup, scheduler=constant).
    """
    if lr_scheduler == "constant" and warmup_steps == 0:
        return None

    try:
        from torch.optim.lr_scheduler import LambdaLR
    except ImportError:
        return None

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # After warmup
        if total_steps <= warmup_steps or total_steps == 0:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(progress, 1.0)
        if lr_scheduler == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if lr_scheduler == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0   # constant after warmup

    return LambdaLR(optimizer, lr_lambda)

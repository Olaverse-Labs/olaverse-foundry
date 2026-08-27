"""Parameter selection shared by the trainers."""
from __future__ import annotations

from typing import Any


def trainable_parameters(module: Any) -> list:
    """
    The parameters an optimizer should actually own.

    Passing frozen parameters to an optimizer mostly works — AdamW skips any
    parameter whose ``.grad`` is ``None``, so no state is allocated for them —
    but it makes a LoRA or partially-frozen run behave unpredictably: gradient
    clipping walks tensors that never have gradients, ``len(param_groups[0])``
    stops describing what is being trained, and anything that later flips
    ``requires_grad`` silently starts updating weights the caller froze.

    Filtering here makes "what trains" explicit and identical across trainers.
    """
    return [p for p in module.parameters() if p.requires_grad]

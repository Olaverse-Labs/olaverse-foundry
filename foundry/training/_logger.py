"""
Optional W&B / TensorBoard logging shim.

Import with:  from foundry.training._logger import _FoundryLogger
"""
from __future__ import annotations


class _FoundryLogger:
    """
    Thin wrapper that unifies wandb and tensorboard behind one interface.

    Both backends are optional — if the requested backend is not installed
    a warning is printed and logging silently becomes a no-op.

    Args:
        backend:  "none" | "wandb" | "tensorboard"
        project:  W&B project name (ignored for tensorboard).
        run_name: Human-readable run label.
        config:   Hyperparameter dict logged as run metadata.
    """

    def __init__(
        self,
        backend:  str,
        project:  str,
        run_name: str,
        config:   dict,
    ) -> None:
        self._backend = backend
        self._handle  = None

        if backend == "wandb":
            try:
                import wandb
                self._handle = wandb.init(
                    project=project,
                    name=run_name or None,
                    config=config,
                    reinit=True,
                )
            except Exception as exc:
                print(f"[foundry] wandb init failed — logging disabled: {exc}")

        elif backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                comment = f"_{run_name}" if run_name else ""
                self._handle = SummaryWriter(comment=comment)
            except Exception as exc:
                print(f"[foundry] tensorboard init failed — logging disabled: {exc}")

    def log(self, step: int, loss: float) -> None:
        if self._handle is None:
            return
        try:
            if self._backend == "wandb":
                self._handle.log({"train/loss": loss}, step=step)
            else:
                self._handle.add_scalar("train/loss", loss, step)
        except Exception:
            pass

    def log_eval(self, step: int, eval_loss: float) -> None:
        if self._handle is None:
            return
        try:
            if self._backend == "wandb":
                self._handle.log({"eval/loss": eval_loss}, step=step)
            else:
                self._handle.add_scalar("eval/loss", eval_loss, step)
        except Exception:
            pass

    def finish(self) -> None:
        if self._handle is None:
            return
        try:
            if self._backend == "wandb":
                self._handle.finish()
            else:
                self._handle.close()
        except Exception:
            pass

from __future__ import annotations

import lightning.pytorch as pl


class TextProgressCallback(pl.Callback):
    def __init__(self, print_every_n_steps: int = 20):
        super().__init__()
        self.print_every_n_steps = int(print_every_n_steps)

    @staticmethod
    def _metric_to_str(value) -> str:
        try:
            return f"{float(value):.6f}"
        except Exception:
            return str(value)

    def on_train_epoch_start(self, trainer, pl_module):
        batches_per_epoch = trainer.num_training_batches
        print(
            f"\n=== Epoch {trainer.current_epoch} started | batches_per_epoch={batches_per_epoch} ===",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.print_every_n_steps != 0:
            return

        batches_per_epoch = trainer.num_training_batches
        batch_in_epoch = batch_idx + 1
        epoch_pct = 100.0 * batch_in_epoch / max(batches_per_epoch, 1)
        max_steps = trainer.max_steps if trainer.max_steps is not None else -1
        optimizer_step = trainer.global_step
        run_pct = 100.0 * optimizer_step / max(max_steps, 1) if max_steps > 0 else 0.0

        train_loss = trainer.callback_metrics.get("train_loss")
        loss_str = self._metric_to_str(train_loss) if train_loss is not None else "n/a"

        print(
            f"[train] epoch={trainer.current_epoch} "
            f"batch={batch_in_epoch}/{batches_per_epoch} "
            f"epoch_progress={epoch_pct:.2f}% "
            f"optimizer_step={optimizer_step}/{max_steps} "
            f"run_progress={run_pct:.2f}% "
            f"train_loss={loss_str}",
            flush=True,
        )

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return

        print(
            f"[val] epoch={trainer.current_epoch} optimizer_step={trainer.global_step} val_loss={self._metric_to_str(val_loss)}",
            flush=True,
        )

import argparse
import os
import random

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from market_simulation.models.utils_vqgan import RawMinuteOrderImageDataset, instantiate_vq_model


REPO_ROOT = Path(__file__).resolve().parents[3]


class OrderImageDataModule(pl.LightningDataModule):
    """Create the train and validation dataloaders for VQGAN training."""
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int,
        num_workers: int,
        converter_json_path: str,
        prefetch_factor: int = 1,
        include_empty_minutes: bool = False,
        max_train_minutes_per_file: int | None = None,
        max_val_minutes_per_file: int | None = None,
        train_minute_stride: int = 1,
        val_minute_stride: int = 1,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.converter_json_path = converter_json_path
        self.prefetch_factor = int(prefetch_factor)
        self.include_empty_minutes = bool(include_empty_minutes)
        self.max_train_minutes_per_file = max_train_minutes_per_file
        self.max_val_minutes_per_file = max_val_minutes_per_file
        self.train_minute_stride = int(train_minute_stride)
        self.val_minute_stride = int(val_minute_stride)
        self._train: RawMinuteOrderImageDataset | None = None
        self._val: RawMinuteOrderImageDataset | None = None

    def setup(self, stage: str | None = None):
        """Instantiate the train and validation datasets."""
        train_files = list(Path(self.train_dir).glob("*messages.parquet"))
        val_files = list(Path(self.val_dir).glob("*messages.parquet"))

        self._train = RawMinuteOrderImageDataset(
            message_files=train_files,
            converter_json_path=self.converter_json_path,
            include_empty_minutes=self.include_empty_minutes,
            max_minutes_per_file=self.max_train_minutes_per_file,
            minute_stride=self.train_minute_stride,
        )
        self._val = RawMinuteOrderImageDataset(
            message_files=val_files,
            converter_json_path=self.converter_json_path,
            include_empty_minutes=self.include_empty_minutes,
            max_minutes_per_file=self.max_val_minutes_per_file,
            minute_stride=self.val_minute_stride,
        )

    def train_dataloader(self):
        """Build the shuffled training dataloader."""
        loader_kwargs = {
            "dataset": self._train,
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            **loader_kwargs,
        )

    def val_dataloader(self):
        """Build the validation dataloader."""
        loader_kwargs = {
            "dataset": self._val,
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_kwargs)


class TextProgressCallback(Callback):
    """Print lightweight text progress during training and validation."""
    def __init__(self, print_every_n_steps: int = 20):
        super().__init__()
        self.print_every_n_steps = int(print_every_n_steps)

    def on_train_epoch_start(self, trainer, pl_module):
        """Print the start of a training epoch."""
        print(
            f"\n=== Epoch {trainer.current_epoch} started | total_batches={trainer.num_training_batches} ===",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Print periodic training progress updates."""
        if trainer.global_step == 0 or trainer.global_step % self.print_every_n_steps != 0:
            return
        max_steps = trainer.max_steps if trainer.max_steps is not None else -1
        print(
            f"[train] epoch={trainer.current_epoch} "
            f"batch={batch_idx + 1}/{trainer.num_training_batches} "
            f"global_step={trainer.global_step}/{max_steps}",
            flush=True,
        )

    def on_validation_end(self, trainer, pl_module):
        """Print the latest validation metrics after each validation run."""
        metrics = trainer.callback_metrics
        for key in ("val/rec_loss", "val/aeloss", "val/disc_loss"):
            value = metrics.get(key)
            if value is not None:
                try:
                    print(
                        f"[val] epoch={trainer.current_epoch} global_step={trainer.global_step} {key}={float(value):.6f}",
                        flush=True,
                    )
                except Exception:
                    print(
                        f"[val] epoch={trainer.current_epoch} global_step={trainer.global_step} {key}={value}",
                        flush=True,
                    )


def main():
    """Parse arguments, build the trainer, and launch fitting."""
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=4.5e-6)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--include_empty_minutes", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_train_minutes_per_file", type=int, default=None)
    p.add_argument("--max_val_minutes_per_file", type=int, default=10)
    p.add_argument("--train_minute_stride", type=int, default=1)
    p.add_argument("--val_minute_stride", type=int, default=1)
    p.add_argument("--precision", default=32, choices=["16", "32", "bf16"])
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium"])
    p.add_argument("--run_root", default=str(REPO_ROOT / "mars_runs" / "vqgan"))
    p.add_argument("--run_name", default=None)
    p.add_argument(
        "--converter_json_path",
        default=str(Path(__file__).resolve().with_name("converters_portable.json")),
    )
    p.add_argument(
        "--vq_config",
        default=str(REPO_ROOT / "third_party" / "latent_diffusion" / "models" / "first_stage_models" / "vq-f4" / "config.yaml"),
    )
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--print_every_n_steps", type=int, default=20)
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic

    dm = OrderImageDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        converter_json_path=args.converter_json_path,
        prefetch_factor=args.prefetch_factor,
        include_empty_minutes=args.include_empty_minutes,
        max_train_minutes_per_file=args.max_train_minutes_per_file,
        max_val_minutes_per_file=args.max_val_minutes_per_file,
        train_minute_stride=args.train_minute_stride,
        val_minute_stride=args.val_minute_stride,
    )

    model = instantiate_vq_model(
        config_path=Path(args.vq_config),
        init_ckpt=args.init_ckpt,
        learning_rate=args.lr,
    )

    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / "tensorboard" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for filename in run_dir.glob("*.ckpt"):
        filename.unlink()

    logger = TensorBoardLogger(
        save_dir=str(run_root),
        name="tensorboard",
        version=run_name,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="step={step}-val_rec_loss={val/rec_loss:.6f}",
        monitor="val/rec_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    progress_cb = TextProgressCallback(print_every_n_steps=args.print_every_n_steps)

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=logger,
        callbacks=[ckpt_cb, progress_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="auto",
        max_steps=args.max_steps,
        precision=args.precision,
        log_every_n_steps=4,
        val_check_interval=120,
        deterministic=args.deterministic,
        enable_checkpointing=True,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

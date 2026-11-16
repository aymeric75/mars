import argparse
import os
import torch
import lightning.pytorch as pl
import numpy as np

from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import BatchSampler, DataLoader

from custom_code.training.utils import TextProgressCallback
from market_simulation.models.order_batch_model import OrderBatchModel
from market_simulation.models.utils_order_batch_model import (
    OnlineMessageTokenDataset,
    VQRuntimeConfig,
    lm_loss_next_token,
)


class TemporalSpacingBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: OnlineMessageTokenDataset,
        batch_size: int,
        num_samples: int | None = None,
        temporal_block_minutes: int = 15,
        min_anchor_spacing_seconds: int = 30,
        chunk_size: int = 256,
        seed: int = 7,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if temporal_block_minutes <= 0:
            raise ValueError("temporal_block_minutes must be positive")
        if min_anchor_spacing_seconds <= 0:
            raise ValueError("min_anchor_spacing_seconds must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.temporal_block_ns = int(temporal_block_minutes) * 60 * 1_000_000_000
        self.min_anchor_spacing_ns = int(min_anchor_spacing_seconds) * 1_000_000_000
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.chunk_groups: list[tuple[int, np.ndarray]] = []
        total_selected = 0

        for file_idx, anchor_times in enumerate(self.dataset.valid_anchor_times):
            if anchor_times.size == 0:
                continue

            block_ids = anchor_times // self.temporal_block_ns
            block_start = 0
            while block_start < anchor_times.size:
                block_end = block_start + 1
                while block_end < anchor_times.size and block_ids[block_end] == block_ids[block_start]:
                    block_end += 1

                spaced_local_indices = []
                last_time = None
                for local_idx in range(block_start, block_end):
                    anchor_time = int(anchor_times[local_idx])
                    if last_time is None or anchor_time - last_time >= self.min_anchor_spacing_ns:
                        spaced_local_indices.append(local_idx)
                        last_time = anchor_time

                for start in range(0, len(spaced_local_indices), self.chunk_size):
                    local_chunk = np.asarray(spaced_local_indices[start : start + self.chunk_size], dtype=np.int64)
                    if local_chunk.size == 0:
                        continue
                    self.chunk_groups.append((file_idx, local_chunk))
                    total_selected += int(local_chunk.size)

                block_start = block_end

        self.total_selected = total_selected
        self.num_samples = self.total_selected if num_samples is None else min(int(num_samples), self.total_selected)

    def __iter__(self):
        if not self.chunk_groups or self.num_samples <= 0:
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        yielded = 0
        batch: list[int] = []
        chunk_order = torch.randperm(len(self.chunk_groups), generator=generator).tolist()

        for pos, chunk_id in enumerate(chunk_order):
            if yielded >= self.num_samples:
                break

            file_idx, local_indices = self.chunk_groups[chunk_id]
            self.dataset.prefetch_chunk(file_idx, local_indices)

            if pos + 1 < len(chunk_order):
                next_file_idx, next_local_indices = self.chunk_groups[chunk_order[pos + 1]]
                self.dataset.prefetch_chunk(next_file_idx, next_local_indices)

            file_base = 0 if file_idx == 0 else self.dataset.cumulative_windows[file_idx - 1]
            local_order = torch.randperm(len(local_indices), generator=generator).tolist()

            for offset in local_order:
                batch.append(file_base + int(local_indices[offset]))
                yielded += 1

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

                if yielded >= self.num_samples:
                    break

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class OrderBatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        pattern: str,
        batch_size: int,
        num_workers: int,
        cache_size: int,
        vq_runtime: VQRuntimeConfig,
        seed: int = 7,
        val_num_samples: int | None = None,
        temporal_block_minutes: int = 15,
        min_anchor_spacing_seconds: int = 30,
        train_chunk_size: int = 256,
        val_chunk_size: int | None = None,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pattern = pattern
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_size = int(cache_size)
        self.vq_runtime = vq_runtime
        self.seed = int(seed)
        self.val_num_samples = None if val_num_samples is None else int(val_num_samples)
        self.temporal_block_minutes = int(temporal_block_minutes)
        self.min_anchor_spacing_seconds = int(min_anchor_spacing_seconds)
        self.train_chunk_size = int(train_chunk_size)
        self.val_chunk_size = self.train_chunk_size if val_chunk_size is None else int(val_chunk_size)
        self._train = None
        self._val = None

    def setup(self, stage: str | None = None):
        train_files = sorted(Path(self.train_dir).glob(self.pattern))
        val_files = sorted(Path(self.val_dir).glob(self.pattern))
        self._train = OnlineMessageTokenDataset(
            message_files=[str(p) for p in train_files],
            cache_size=self.cache_size,
            vq_runtime=self.vq_runtime,
        )
        self._val = OnlineMessageTokenDataset(
            message_files=[str(p) for p in val_files],
            cache_size=self.cache_size,
            vq_runtime=self.vq_runtime,
        )

    def train_dataloader(self):
        batch_sampler = TemporalSpacingBatchSampler(
            self._train,
            batch_size=self.batch_size,
            temporal_block_minutes=self.temporal_block_minutes,
            min_anchor_spacing_seconds=self.min_anchor_spacing_seconds,
            chunk_size=self.train_chunk_size,
            seed=self.seed,
            drop_last=True,
        )
        return DataLoader(
            self._train,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        batch_sampler = TemporalSpacingBatchSampler(
            self._val,
            batch_size=self.batch_size,
            num_samples=self.val_num_samples,
            temporal_block_minutes=self.temporal_block_minutes,
            min_anchor_spacing_seconds=self.min_anchor_spacing_seconds,
            chunk_size=self.val_chunk_size,
            seed=self.seed,
            drop_last=False,
        )
        return DataLoader(
            self._val,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class OrderBatchLightningModule(pl.LightningModule):
    def __init__(self, emb_dim: int, num_layers: int, num_heads: int, vocab_size: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = OrderBatchModel(
            emb_dim=int(emb_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            vocab_size=int(vocab_size),
        )
        self.lr = float(lr)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch
        logits = self(input_ids)
        loss = lm_loss_next_token(logits, input_ids)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        logits = self(input_ids)
        loss = lm_loss_next_token(logits, input_ids)
        self.log("val_loss", loss, on_step=False, sync_dist=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class PrintCkptFilename(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                fp = cb.format_checkpoint_name(metrics=trainer.callback_metrics, filename=cb.filename)
                full = fp if os.path.isabs(fp) else os.path.join(cb.dirpath, fp)
                print(" >> checkpoint dirpath:", cb.dirpath)
                print(" >> checkpoint full path:", full)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", default="checkpoints_batch_order_model")
    p.add_argument("--run_name", default=None)
    p.add_argument("--train_dir", default="/scratch/project_2012747/mars_data/train")
    p.add_argument("--val_dir", default="/scratch/project_2012747/mars_data/val")
    p.add_argument("--pattern", default="*messages*.parquet")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cache_size", type=int, default=2)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_num_samples", type=int, default=None)
    p.add_argument("--temporal_block_minutes", type=int, default=15)
    p.add_argument("--min_anchor_spacing_seconds", type=int, default=30)
    p.add_argument("--train_chunk_size", type=int, default=256)
    p.add_argument("--val_chunk_size", type=int, default=None)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--latent_diffusion_root", default="../../../third_party/latent_diffusion")
    p.add_argument("--taming_root", default="../../../third_party/taming-transformers")
    p.add_argument("--vq_ckpt_dir", default="./")
    args = p.parse_args()

    run_root = args.run_root
    os.makedirs(run_root, exist_ok=True)
    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"
    run_dir = os.path.join(run_root, "tensorboard", run_name)
    os.makedirs(run_dir, exist_ok=True)

    for filename in os.listdir(run_dir):
        if filename.endswith(".ckpt"):
            os.remove(os.path.join(run_dir, filename))

    print("PWD =", os.getcwd())
    print("run_root =", os.path.abspath(run_root))
    print("run_name =", run_name)
    print("run_dir =", os.path.abspath(run_dir))

    pl.seed_everything(args.seed, workers=True)

    vq_runtime = VQRuntimeConfig(
        ckpt_dir=args.vq_ckpt_dir,
        latent_diffusion_root=args.latent_diffusion_root,
        taming_root=args.taming_root,
    )

    dm = OrderBatchDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        pattern=args.pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        vq_runtime=vq_runtime,
        seed=args.seed,
        val_num_samples=args.val_num_samples,
        temporal_block_minutes=args.temporal_block_minutes,
        min_anchor_spacing_seconds=args.min_anchor_spacing_seconds,
        train_chunk_size=args.train_chunk_size,
        val_chunk_size=args.val_chunk_size,
    )

    model = OrderBatchLightningModule(
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        lr=args.lr,
    )

    logger = TensorBoardLogger(save_dir=run_root, name="tensorboard", version=run_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    progress_cb = TextProgressCallback(print_every_n_steps=5)

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[ckpt_cb, PrintCkptFilename(), progress_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="auto",
        max_steps=args.max_steps,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=50,
        limit_val_batches=10,
        enable_checkpointing=True,
        enable_progress_bar=False,
        deterministic=True,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

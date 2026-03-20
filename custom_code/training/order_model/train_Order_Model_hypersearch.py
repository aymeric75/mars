import argparse
import os
import random
import glob
import lightning.pytorch as pl
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import BatchSampler, DataLoader, Subset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from market_simulation.models.utils_order_model import (
    RawMessagesTokenDataset,
    build_model_from_variant,
    lm_loss_all_positions
)


class ChunkShuffleBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: RawMessagesTokenDataset,
        batch_size: int,
        num_samples: int | None = None,
        chunk_size: int = 2048,
        seed: int = 7,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.chunk_starts: list[tuple[int, int, int]] = []
        prev_total = 0
        for file_idx, file_windows in enumerate(self.dataset.window_counts):
            for start in range(0, file_windows, self.chunk_size):
                chunk_len = min(self.chunk_size, file_windows - start)
                self.chunk_starts.append((file_idx, start, chunk_len))
            prev_total += file_windows

    def __iter__(self):
        if not self.chunk_starts or self.num_samples <= 0:
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        yielded = 0
        chunk_order = torch.randperm(len(self.chunk_starts), generator=generator).tolist()
        batch: list[int] = []

        for pos, chunk_id in enumerate(chunk_order):
            if yielded >= self.num_samples:
                break

            file_idx, start, chunk_len = self.chunk_starts[chunk_id]
            self.dataset.prefetch_file(file_idx)

            if pos + 1 < len(chunk_order):
                next_chunk_id = chunk_order[pos + 1]
                next_file_idx = self.chunk_starts[next_chunk_id][0]
                if next_file_idx != file_idx:
                    self.dataset.prefetch_file(next_file_idx)

            file_base = 0 if file_idx == 0 else self.dataset.cumulative_windows[file_idx - 1]
            local_order = torch.randperm(chunk_len, generator=generator).tolist()

            for offset in local_order:
                batch.append(file_base + start + offset)
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
        batch_size: int,
        num_workers: int,
        seed: int = 7,
        pattern: str = "*_messages.parquet",
        feature_cols: int = 15,
        seq_len: int = 1024,
        cache_size: int = 2,
        shuffle_val: bool = False,
        train_num_samples: int | None = None,
        train_chunk_size: int = 2048,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pattern = pattern
        self.feature_cols = int(feature_cols)
        self.seed = seed
        self.seq_len = int(seq_len)
        self.cache_size = int(cache_size)
        self.shuffle_val = bool(shuffle_val)
        self.train_num_samples = None if train_num_samples is None else int(train_num_samples)
        self.train_chunk_size = int(train_chunk_size)
        self._train = None
        self._val = None



    def setup(self, stage: str | None = None):
        self._train = RawMessagesTokenDataset(
            message_files=list(Path(self.train_dir).glob("*messages.parquet")),
            seq_len=self.seq_len,
            cache_size=self.cache_size,
        )
        self._val = RawMessagesTokenDataset(
            message_files=list(Path(self.val_dir).glob("*messages.parquet")),
            seq_len=self.seq_len,
            cache_size=self.cache_size,
        )
    
    def collate_tokens(self, batch):
        batch = [x.contiguous().clone() for x in batch]
        return torch.stack(batch, dim=0)



    @staticmethod
    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def train_dataloader(self):
        batch_sampler = ChunkShuffleBatchSampler(
            self._train,
            batch_size=self.batch_size,
            num_samples=self.train_num_samples,
            chunk_size=self.train_chunk_size,
            seed=self.seed,
            drop_last=True,
        )

        return DataLoader(
            self._train,
            shuffle=False,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_tokens,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            collate_fn=self.collate_tokens,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            generator=g,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )



class OrderLightningModule(pl.LightningModule):
    def __init__(self, model_variant: str, K: int, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.model, _ = build_model_from_variant(model_variant, K=int(K))
        self.lr = float(lr)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X = batch

        if torch.cuda.is_available() and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()
        
        logits = self(X)
        loss = lm_loss_all_positions(logits, X)
    
        if torch.cuda.is_available() and batch_idx == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**2
            reserved = torch.cuda.max_memory_reserved() / 1024**2
            print(f"\nGPU memory — allocated: {peak:.1f} MB | reserved: {reserved:.1f} MB\n")
        
        self.log(
            "train_loss",
            loss,
            on_step=True,
            sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch
        logits = self(X)
        loss = lm_loss_all_positions(logits, X)
        self.log("val_loss", loss, on_step=False, sync_dist=False, prog_bar=False)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)



class TextProgressCallback(Callback):
    def __init__(self, print_every_n_steps: int = 20):
        super().__init__()
        self.print_every_n_steps = int(print_every_n_steps)

    def on_train_epoch_start(self, trainer, pl_module):
        total_batches = trainer.num_training_batches
        print(f"\n=== Epoch {trainer.current_epoch} started | total_batches={total_batches} ===", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0:
            return

        if trainer.global_step % self.print_every_n_steps != 0:
            return

        total_batches = trainer.num_training_batches
        batch_in_epoch = (batch_idx + 1)
        epoch_pct = 100.0 * batch_in_epoch / max(total_batches, 1)

        max_steps = trainer.max_steps if trainer.max_steps is not None else -1
        step_pct = 100.0 * trainer.global_step / max(max_steps, 1) if max_steps > 0 else 0.0
        batch_size = len(batch)
        total_windows = len(trainer.train_dataloader.dataset)
        sampled_windows = trainer.global_step * batch_size
        sampled_pct = 100.0 * sampled_windows / max(total_windows, 1)

        print(
            f"[train] epoch={trainer.current_epoch} "
            f"batch={batch_in_epoch}/{total_batches} "
            f"epoch_progress={epoch_pct:.2f}% "
            f"global_step={trainer.global_step}/{max_steps} "
            f"step_progress={step_pct:.2f}% "
            f"sampled_windows={sampled_windows}/{total_windows} "
            f"sampled_progress={sampled_pct:.5f}%",
            flush=True,
        )

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss", None)
        if val_loss is not None:
            try:
                val_loss = float(val_loss)
                print(f"[val] epoch={trainer.current_epoch} global_step={trainer.global_step} val_loss={val_loss:.6f}", flush=True)
            except Exception:
                print(f"[val] epoch={trainer.current_epoch} global_step={trainer.global_step} val_loss={val_loss}", flush=True)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--pattern", default="*_messages.parquet")
    p.add_argument("--feature_cols", type=int, default=15)
    p.add_argument("--cache_size", type=int, default=8)
    p.add_argument("--shuffle_val", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train_num_samples", type=int, default=None)
    p.add_argument("--train_chunk_size", type=int, default=2048)
    p.add_argument("--model_variant", default="base", choices=["base", "small"])
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium"])
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--run_root", default="checkpoints_order_model")
    p.add_argument("--run_name", default=None)

    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic

    dm = OrderBatchDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pattern=args.pattern,
        feature_cols=args.feature_cols,
        seq_len=args.K,
        cache_size=args.cache_size,
        shuffle_val=args.shuffle_val,
        train_num_samples=args.train_num_samples,
        train_chunk_size=args.train_chunk_size,
    )

    model = OrderLightningModule(model_variant=args.model_variant, K=args.K, lr=args.lr)


    run_dir = args.run_root
    os.makedirs(run_dir, exist_ok=True)

    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"

    logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tensorboard",
        version=run_name,   # <--- makes each run distinct in one TB logdir
    )



    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,                          # <--- checkpoints saved HERE
        filename="step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    progress_cb = TextProgressCallback(print_every_n_steps=20)

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[ckpt_cb, progress_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="auto",
        max_steps=args.max_steps,
        precision=args.precision,
        log_every_n_steps=4,
        val_check_interval=120,
        limit_val_batches=10,
        deterministic=args.deterministic,
        enable_checkpointing=True,
        enable_progress_bar=False,
    )


    trainer.fit(model, dm)


if __name__ == "__main__":
    # Lightning uses this env var in some cluster setups; harmless otherwise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

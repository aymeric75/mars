from __future__ import annotations

import argparse
import os

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import BatchSampler, DataLoader

from custom_code.training.order_batch_model.train_Order_Batch_Model_hypersearch import (
    OrderBatchLightningModule,
)
from custom_code.training.order_model.train_Order_Model_hypersearch import (
    OrderLightningModule,
)
from custom_code.training.utils import TextProgressCallback
from market_simulation.models.heads import ReturnHeads
from market_simulation.models.utils_heads import (
    OnlineReturnHeadDataset,
    VQRuntimeConfig,
    buy_fill_price,
    sell_fill_price,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLASS_LABELS = ["unprofitable", "unclear", "profitable"]
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"


class HeadsChunkBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: OnlineReturnHeadDataset,
        batch_size: int,
        num_samples: int | None = None,
        chunk_size: int = 256,
        seed: int = 7,
        drop_last: bool = True,
        resample_each_iter: bool = True,
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
        self.resample_each_iter = bool(resample_each_iter)
        self._iteration = 0

        self.chunk_starts: list[tuple[int, int, int]] = []
        for file_idx, file_windows in enumerate(self.dataset.window_counts):
            for start in range(0, file_windows, self.chunk_size):
                chunk_len = min(self.chunk_size, file_windows - start)
                self.chunk_starts.append((file_idx, start, chunk_len))

    def __iter__(self):
        if not self.chunk_starts or self.num_samples <= 0:
            return

        generator = torch.Generator()
        if self.resample_each_iter:
            generator.manual_seed(self.seed + self._iteration)
            self._iteration += 1
        else:
            generator.manual_seed(self.seed)

        yielded = 0
        batch: list[int] = []
        chunk_order = torch.randperm(len(self.chunk_starts), generator=generator).tolist()

        for pos, chunk_id in enumerate(chunk_order):
            if yielded >= self.num_samples:
                break

            file_idx, start, chunk_len = self.chunk_starts[chunk_id]
            local_positions = np.arange(start, start + chunk_len, dtype=np.int64)
            self.dataset.prefetch_chunk(file_idx, local_positions)

            if pos + 1 < len(chunk_order):
                next_chunk_id = chunk_order[pos + 1]
                next_file_idx, next_start, next_chunk_len = self.chunk_starts[next_chunk_id]
                next_positions = np.arange(next_start, next_start + next_chunk_len, dtype=np.int64)
                self.dataset.prefetch_chunk(next_file_idx, next_positions)

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


class HeadsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_dir: str,
        val_dir: str,
        pattern: str,
        scenario: str,
        seq_len: int,
        horizon_seconds: int,
        batch_size: int,
        num_workers: int,
        cache_size: int,
        train_num_samples: int | None,
        train_chunk_size: int,
        val_num_samples: int | None,
        val_chunk_size: int | None,
        seed: int,
        vq_runtime: VQRuntimeConfig | None,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pattern = pattern
        self.scenario = scenario
        self.seq_len = int(seq_len)
        self.horizon_seconds = int(horizon_seconds)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_size = int(cache_size)
        self.train_num_samples = None if train_num_samples is None else int(train_num_samples)
        self.train_chunk_size = int(train_chunk_size)
        self.val_num_samples = None if val_num_samples is None else int(val_num_samples)
        self.val_chunk_size = self.train_chunk_size if val_chunk_size is None else int(val_chunk_size)
        self.seed = int(seed)
        self.vq_runtime = vq_runtime
        self.train_ds: OnlineReturnHeadDataset | None = None
        self.val_ds: OnlineReturnHeadDataset | None = None

    def setup(self, stage: str | None = None):
        if self.train_ds is not None and self.val_ds is not None:
            return
        train_files = sorted(Path(self.train_dir).glob(self.pattern))
        val_files = sorted(Path(self.val_dir).glob(self.pattern))
        self.train_ds = OnlineReturnHeadDataset(
            message_files=[str(path) for path in train_files],
            seq_len=self.seq_len,
            scenario=self.scenario,
            horizon_seconds=self.horizon_seconds,
            cache_size=self.cache_size,
            feature_chunk_size=self.train_chunk_size,
            sample_chunk_size=self.train_chunk_size,
            vq_runtime=self.vq_runtime,
        )
        self.val_ds = OnlineReturnHeadDataset(
            message_files=[str(path) for path in val_files],
            seq_len=self.seq_len,
            scenario=self.scenario,
            horizon_seconds=self.horizon_seconds,
            cache_size=self.cache_size,
            feature_chunk_size=self.val_chunk_size,
            sample_chunk_size=self.val_chunk_size,
            vq_runtime=self.vq_runtime,
        )

    @staticmethod
    def collate_batch(batch):
        keys = batch[0].keys()
        return {key: torch.stack([sample[key].contiguous().clone() for sample in batch], dim=0) for key in keys}

    def train_dataloader(self):
        batch_sampler = HeadsChunkBatchSampler(
            self.train_ds,
            batch_size=self.batch_size,
            num_samples=self.train_num_samples,
            chunk_size=self.train_chunk_size,
            seed=self.seed,
            drop_last=True,
            resample_each_iter=True,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        batch_sampler = HeadsChunkBatchSampler(
            self.val_ds,
            batch_size=self.batch_size,
            num_samples=self.val_num_samples,
            chunk_size=self.val_chunk_size,
            seed=self.seed,
            drop_last=False,
            resample_each_iter=False,
        )
        return DataLoader(
            self.val_ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class HeadsLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        head_type: str,
        scenario: str,
        trade_side: str,
        trade_quantity: float,
        order_model_ckpt: str | None,
        order_batch_ckpt: str | None,
        hidden_dim: int,
        dropout: float,
        lr: float,
        pnl_margin: float = 0.0,
        regression_loss_weight: float = 1.0,
        probability_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        order_feature_dim: int | None = None
        batch_feature_dim: int | None = None

        self.order_model = None
        if scenario in {"order_model", "both"}:
            if not order_model_ckpt:
                raise ValueError("order_model_ckpt is required for scenario using the Order Model")
            order_lm = OrderLightningModule.load_from_checkpoint(order_model_ckpt, map_location="cpu", strict=True)
            self.order_model = order_lm.model
            self.order_model.eval()
            for param in self.order_model.parameters():
                param.requires_grad_(False)
            order_feature_dim = int(self.order_model.emb_dim)

        self.order_batch_model = None
        if scenario in {"order_batch", "both"}:
            if not order_batch_ckpt:
                raise ValueError("order_batch_ckpt is required for scenario using the Order-Batch Model")
            batch_lm = OrderBatchLightningModule.load_from_checkpoint(order_batch_ckpt, map_location="cpu", strict=True)
            self.order_batch_model = batch_lm.model
            self.order_batch_model.eval()
            for param in self.order_batch_model.parameters():
                param.requires_grad_(False)
            batch_feature_dim = int(self.order_batch_model.decoder.config.hidden_size)

        self.model = ReturnHeads(
            order_feature_dim=order_feature_dim,
            batch_feature_dim=batch_feature_dim,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.lr = float(lr)
        self.register_buffer("probability_class_weights", torch.ones(3, dtype=torch.float32), persistent=False)

    def _pnl_targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bid_prices = batch["bid_prices"].tolist()
        ask_prices = batch["ask_prices"].tolist()
        bid_sizes = batch["bid_sizes"].tolist()
        ask_sizes = batch["ask_sizes"].tolist()
        future_bid_prices = batch["future_bid_prices"].tolist()
        future_ask_prices = batch["future_ask_prices"].tolist()
        future_bid_sizes = batch["future_bid_sizes"].tolist()
        future_ask_sizes = batch["future_ask_sizes"].tolist()

        pnls = []
        for (
            sample_bid_prices,
            sample_ask_prices,
            sample_bid_sizes,
            sample_ask_sizes,
            sample_future_bid_prices,
            sample_future_ask_prices,
            sample_future_bid_sizes,
            sample_future_ask_sizes,
        ) in zip(
            bid_prices,
            ask_prices,
            bid_sizes,
            ask_sizes,
            future_bid_prices,
            future_ask_prices,
            future_bid_sizes,
            future_ask_sizes,
        ):
            if self.hparams.trade_side == "long":
                entry_fill = buy_fill_price(
                    ask_prices=sample_ask_prices,
                    ask_sizes=sample_ask_sizes,
                    quantity=float(self.hparams.trade_quantity),
                )
                exit_fill = sell_fill_price(
                    bid_prices=sample_future_bid_prices,
                    bid_sizes=sample_future_bid_sizes,
                    quantity=float(self.hparams.trade_quantity),
                )
                pnls.append(float(exit_fill - entry_fill))
            else:
                entry_fill = sell_fill_price(
                    bid_prices=sample_bid_prices,
                    bid_sizes=sample_bid_sizes,
                    quantity=float(self.hparams.trade_quantity),
                )
                exit_fill = buy_fill_price(
                    ask_prices=sample_future_ask_prices,
                    ask_sizes=sample_future_ask_sizes,
                    quantity=float(self.hparams.trade_quantity),
                )
                pnls.append(float(entry_fill - exit_fill))

        return torch.tensor(pnls, dtype=torch.float32, device=self.device)

    def _profit_targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pnl = self._pnl_targets(batch)
        margin = float(self.hparams.pnl_margin)
        targets = torch.full_like(pnl, fill_value=1, dtype=torch.long)
        targets = torch.where(pnl < -margin, torch.zeros_like(targets), targets)
        targets = torch.where(pnl > margin, torch.full_like(targets, fill_value=2), targets)
        return targets

    def _encode_features(self, batch: dict[str, torch.Tensor]) -> tuple[Tensor | None, Tensor | None]:
        order_features = None
        batch_features = None

        if self.order_model is not None:
            order_context = batch["order_context"].to(device=self.device, dtype=torch.long, non_blocking=True)
            with torch.no_grad():
                self.order_model.eval()
                order_features = self.order_model.encode(order_context)

        if self.order_batch_model is not None:
            batch_tokens = batch["batch_tokens"].to(device=self.device, dtype=torch.long, non_blocking=True)
            batch_tokens = batch_tokens.view(batch_tokens.size(0), -1)
            with torch.no_grad():
                self.order_batch_model.eval()
                batch_features = self.order_batch_model.encode(batch_tokens)

        return order_features, batch_features

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        target_return = batch["target_return"].to(device=self.device, dtype=torch.float32, non_blocking=True)
        order_features, batch_features = self._encode_features(batch)
        outputs = self.model(order_features=order_features, batch_features=batch_features)

        total_loss = torch.zeros((), device=self.device)
        on_step = stage == "train"

        if self.hparams.head_type in {"regression", "multitask"}:
            regression_loss = F.smooth_l1_loss(outputs["pred_return"], target_return)
            regression_mae = torch.mean(torch.abs(outputs["pred_return"] - target_return))
            total_loss = total_loss + float(self.hparams.regression_loss_weight) * regression_loss
            self.log(f"{stage}_regression_loss", regression_loss, on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_regression_mae", regression_mae, on_step=on_step, on_epoch=True, sync_dist=True)

        if self.hparams.head_type == "pnl_regression":
            target_pnl = self._pnl_targets(batch)
            pnl_loss = F.smooth_l1_loss(outputs["pred_pnl"], target_pnl)
            pnl_mae = torch.mean(torch.abs(outputs["pred_pnl"] - target_pnl))
            total_loss = total_loss + pnl_loss
            self.log(f"{stage}_pnl_loss", pnl_loss, on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_pnl_mae", pnl_mae, on_step=on_step, on_epoch=True, sync_dist=True)

        if self.hparams.head_type in {"probability", "multitask"}:
            profit_target = self._profit_targets(batch)
            probability_loss = F.cross_entropy(
                outputs["profit_logits"],
                profit_target,
                weight=self.probability_class_weights,
            )
            probability_acc = torch.mean((torch.argmax(outputs["profit_logits"], dim=-1) == profit_target).to(dtype=torch.float32))
            total_loss = total_loss + float(self.hparams.probability_loss_weight) * probability_loss
            self.log(f"{stage}_probability_loss", probability_loss, on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_probability_acc", probability_acc, on_step=on_step, on_epoch=True, sync_dist=True)

        self.log(f"{stage}_loss", total_loss, on_step=on_step, on_epoch=True, sync_dist=True, prog_bar=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


def print_probability_class_balance(
    loader: DataLoader,
    model: HeadsLightningModule,
    split: str,
    max_samples: int | None = None,
) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.long)
    seen = 0

    for batch in loader:
        targets = model._profit_targets(batch).detach().cpu()
        if max_samples is not None:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            targets = targets[:remaining]
        counts += torch.bincount(targets, minlength=3)
        seen += int(targets.numel())
        if max_samples is not None and seen >= max_samples:
            break

    rates = counts.to(dtype=torch.float32) / max(seen, 1)
    lines = [f"{split} probability-class balance over {seen} samples"]
    lines.extend(f"{split} {label}={int(counts[idx])} ({float(rates[idx]):.6f})" for idx, label in enumerate(CLASS_LABELS))
    print()
    print(f"{ANSI_BLUE}{'=' * 56}")
    for line in lines:
        print(line)
    print(f"{'=' * 56}{ANSI_RESET}")
    print()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--pattern", default="*messages*.parquet")
    parser.add_argument("--run_root", default="")
    parser.add_argument("--run_name", default=None)

    parser.add_argument(
        "--head_type",
        default="regression",
        choices=["regression", "pnl_regression", "probability", "multitask"],
    )
    parser.add_argument("--scenario", default="order_model", choices=["order_model", "order_batch", "both"])
    parser.add_argument("--trade_side", default="long", choices=["long", "short"])
    parser.add_argument("--trade_quantity", type=float, default=1.0)

    parser.add_argument("--order_model_ckpt", default=None)
    parser.add_argument("--order_batch_ckpt", default=None)
    parser.add_argument("--vq_ckpt_dir", default=None)
    parser.add_argument("--latent_diffusion_root", default=str(REPO_ROOT / "third_party" / "latent_diffusion"))
    parser.add_argument("--taming_root", default=str(REPO_ROOT / "third_party" / "taming-transformers"))
    parser.add_argument(
        "--vq_config_relpath",
        default="latent_diffusion/models/first_stage_models/vq-f4/config.yaml",
    )
    parser.add_argument("--vq_use_autocast", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--horizon_seconds", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pnl_margin", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--train_samples_per_epoch", type=int, default=None)
    parser.add_argument("--train_samples_per_val", dest="train_samples_per_epoch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--train_chunk_size", type=int, default=128)
    parser.add_argument("--val_num_samples", type=int, default=256)
    parser.add_argument("--val_chunk_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    parser.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--gradient_clip_algorithm", default="norm", choices=["norm", "value"])
    parser.add_argument("--print_every_n_steps", type=int, default=10)
    parser.add_argument("--regression_loss_weight", type=float, default=1.0)
    parser.add_argument("--probability_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.scenario in {"order_model", "both"} and not args.order_model_ckpt:
        raise ValueError("--order_model_ckpt is required for this scenario")
    if args.scenario in {"order_batch", "both"} and not args.order_batch_ckpt:
        raise ValueError("--order_batch_ckpt is required for this scenario")
    if args.scenario in {"order_batch", "both"} and not args.vq_ckpt_dir:
        raise ValueError("--vq_ckpt_dir is required for this scenario")

    if args.train_samples_per_epoch is not None and args.train_samples_per_epoch <= 0:
        raise ValueError("--train_samples_per_epoch must be positive when provided")

    train_samples_per_epoch = args.batch_size * 1000 if args.train_samples_per_epoch is None else int(args.train_samples_per_epoch)
    train_batches_per_epoch = max(1, train_samples_per_epoch // args.batch_size)
    train_num_samples = train_batches_per_epoch * args.batch_size
    val_check_interval = train_batches_per_epoch

    run_root = args.run_root or str(REPO_ROOT / "mars_runs" / "heads")
    os.makedirs(run_root, exist_ok=True)
    run_name = args.run_name or (f"head={args.head_type}_scenario={args.scenario}_side={args.trade_side}_bs={args.batch_size}_lr={args.lr:g}")
    run_dir = os.path.join(run_root, "tensorboard", run_name)
    os.makedirs(run_dir, exist_ok=True)

    for filename in os.listdir(run_dir):
        if filename.endswith(".ckpt"):
            os.remove(os.path.join(run_dir, filename))

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic

    vq_runtime = None
    if args.scenario in {"order_batch", "both"}:
        vq_runtime = VQRuntimeConfig(
            ckpt_dir=args.vq_ckpt_dir,
            latent_diffusion_root=args.latent_diffusion_root,
            taming_root=args.taming_root,
            config_relpath=args.vq_config_relpath,
            use_autocast=args.vq_use_autocast,
        )

    dm = HeadsDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        pattern=args.pattern,
        scenario=args.scenario,
        seq_len=args.seq_len,
        horizon_seconds=args.horizon_seconds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        train_num_samples=train_num_samples,
        train_chunk_size=args.train_chunk_size,
        val_num_samples=args.val_num_samples,
        val_chunk_size=args.val_chunk_size,
        seed=args.seed,
        vq_runtime=vq_runtime,
    )

    model = HeadsLightningModule(
        head_type=args.head_type,
        scenario=args.scenario,
        trade_side=args.trade_side,
        trade_quantity=args.trade_quantity,
        order_model_ckpt=args.order_model_ckpt,
        order_batch_ckpt=args.order_batch_ckpt,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        pnl_margin=args.pnl_margin,
        regression_loss_weight=args.regression_loss_weight,
        probability_loss_weight=args.probability_loss_weight,
    )

    logger = TensorBoardLogger(save_dir=run_root, name="tensorboard", version=run_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="step={step}-val_loss={val_loss:.8e}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    progress_cb = TextProgressCallback(print_every_n_steps=args.print_every_n_steps)

    if args.head_type in {"probability", "multitask"}:
        dm.setup()
        train_counts = print_probability_class_balance(dm.train_dataloader(), model, split="train", max_samples=4096)
        print_probability_class_balance(dm.val_dataloader(), model, split="val", max_samples=None)
        exit()
        train_counts_f = train_counts.to(dtype=torch.float32)
        class_weights = train_counts_f.sum() / (len(CLASS_LABELS) * train_counts_f.clamp_min(1.0))
        class_weights = class_weights / class_weights.mean()
        model.probability_class_weights.copy_(class_weights)
        print("probability class weights=" + ", ".join(f"{label}={float(weight):.6f}" for label, weight in zip(CLASS_LABELS, class_weights.tolist())))

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[ckpt_cb, progress_cb],
        enable_checkpointing=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="auto",
        max_steps=args.max_steps,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        log_every_n_steps=4,
        val_check_interval=val_check_interval,
        deterministic=args.deterministic,
        enable_progress_bar=False,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

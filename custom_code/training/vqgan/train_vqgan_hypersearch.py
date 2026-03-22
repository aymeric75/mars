import argparse
import inspect
import json
import os
import random
import sys

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from market_simulation.utils.bin_converter import BinConverter


REPO_ROOT = Path(__file__).resolve().parents[3]
LATENT_DIFFUSION_ROOT = REPO_ROOT / "third_party" / "latent_diffusion"
TAMING_ROOT = REPO_ROOT / "third_party" / "taming-transformers"

for extra_path in (LATENT_DIFFUSION_ROOT, TAMING_ROOT):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from ldm.util import instantiate_from_config


NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
ORDER_IMAGE_MAX_VALUE = 100
ONE_MINUTE_NS = 60 * 1_000_000_000
MARKET_OPEN_NS = (9 * 60 * 60 + 30 * 60) * 1_000_000_000
MARKET_CLOSE_NS = (16 * 60 * 60) * 1_000_000_000


@dataclass
class ImageConverters:
    """Holds the bin converters used for order-image construction."""
    price_level: BinConverter
    order_volume: BinConverter


def _build_converters_from_json(converter_json_path: Path) -> ImageConverters:
    """Load image bin converters from a portable JSON file."""
    with converter_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    price_minus_mid: list[float] = []
    for bin_item in obj["state"]["price_level"]["bin_values"]:
        price_minus_mid.extend(bin_item["data"])

    sizes: list[float] = []
    for bin_item in obj["state"]["order_volume"]["bin_values"]:
        sizes.extend(bin_item["data"])

    price_level = BinConverter.create_from_values(price_minus_mid, NUM_BINS_PRICE_LEVEL)
    order_volume = BinConverter.create_from_values(sizes, NUM_BINS_ORDER_VOLUME)
    return ImageConverters(price_level=price_level, order_volume=order_volume)


def _find_default_converter_json() -> Path:
    """Return the first local converters JSON file that exists."""
    candidates = [
        REPO_ROOT / "custom_code" / "preprocessing" / "converters_portable.json",
        REPO_ROOT / "custom_code" / "preprocessing" / "order_model" / "converters_portable.json",
        REPO_ROOT / "custom_code" / "training" / "order_model" / "converters_portable.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a local converters_portable.json file")


def create_mars_order_type_column(messages: pd.DataFrame) -> None:
    """Map raw message types and directions to the 3 MarS order classes."""
    messages["Mars_type"] = 0
    messages.loc[messages["Message_Type"].isin([2, 3]), "Mars_type"] = 2
    messages.loc[
        ((messages["Message_Type"] == 1) & (messages["Direction"] == -1))
        | ((messages["Message_Type"] == 4) & (messages["Direction"] == 1)),
        "Mars_type",
    ] = 1


def from_messages_and_snapshots_to_image_features(
    messages: pd.DataFrame,
    snapshots: pd.DataFrame,
    converters: ImageConverters,
) -> pd.DataFrame:
    """Convert raw messages and snapshots into per-order image features."""
    messages = messages.copy()
    create_mars_order_type_column(messages)

    features = messages[["Time", "Mars_type", "Price", "Size"]].copy()
    features["mid_price"] = (snapshots["Ask_Price_1"] + snapshots["Bid_Price_1"]) / 2
    features = features[
        (features["Time"] >= MARKET_OPEN_NS) & (features["Time"] <= MARKET_CLOSE_NS)
    ].reset_index(drop=True)

    price_minus_mid = (features["Price"] - features["mid_price"]).to_numpy(dtype=np.float64, copy=False)
    size_values = features["Size"].to_numpy(dtype=np.float64, copy=False)

    features["bin_price"] = converters.price_level.get_bin_indices(price_minus_mid).astype(np.int32)
    features["bin_vol"] = converters.order_volume.get_bin_indices(size_values).astype(np.int32)

    return features[["Time", "Mars_type", "bin_price", "bin_vol"]]


def minute_chunk_to_order_image(chunk: pd.DataFrame) -> np.ndarray:
    """Rasterize one minute of orders into a MarS-style RGB image."""
    image = np.zeros((3, NUM_BINS_PRICE_LEVEL, NUM_BINS_ORDER_VOLUME), dtype=np.uint8)
    if chunk.empty:
        return image

    mars_type = np.clip(chunk["Mars_type"].to_numpy(dtype=np.int64, copy=False), 0, 2)
    bin_price = np.clip(
        chunk["bin_price"].to_numpy(dtype=np.int64, copy=False),
        0,
        NUM_BINS_PRICE_LEVEL - 1,
    )
    bin_vol = np.clip(
        chunk["bin_vol"].to_numpy(dtype=np.int64, copy=False),
        0,
        NUM_BINS_ORDER_VOLUME - 1,
    )

    np.add.at(image, (mars_type, bin_price, bin_vol), 1)
    np.clip(image, 0, ORDER_IMAGE_MAX_VALUE, out=image)
    return image


class RawMinuteOrderImageDataset(Dataset):
    """Build one-minute order images on the fly from raw parquet files."""
    def __init__(
        self,
        message_files: list[str | Path],
        converter_json_path: str | Path,
        cache_size: int = 2,
        include_empty_minutes: bool = False,
        max_minutes_per_file: int | None = None,
        minute_stride: int = 1,
    ):
        self.cache_size = int(cache_size)
        self.include_empty_minutes = bool(include_empty_minutes)
        self.max_minutes_per_file = None if max_minutes_per_file is None else int(max_minutes_per_file)
        self.minute_stride = max(1, int(minute_stride))
        self.cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self.index: list[tuple[int, int, int, int]] = []

        self.converters = _build_converters_from_json(Path(converter_json_path))
        self.message_files: list[Path] = []
        self.snapshot_files: list[Path] = []

        for msg_path in sorted(Path(p) for p in message_files):
            snap_path = Path(str(msg_path).replace("messages", "snapshots"))
            if snap_path.exists():
                self.message_files.append(msg_path)
                self.snapshot_files.append(snap_path)
            else:
                print(f"Skipping {msg_path} (no snapshot file)")

        print("Building minute-image dataset index...")
        for file_idx, msg_path in enumerate(self.message_files):
            minute_ranges = self._build_minute_ranges(msg_path)
            if self.minute_stride > 1:
                minute_ranges = minute_ranges[:: self.minute_stride]
            if self.max_minutes_per_file is not None:
                minute_ranges = minute_ranges[: self.max_minutes_per_file]

            self.index.extend((file_idx, minute_id, start, end) for minute_id, start, end in minute_ranges)

        print(f"Total minute images: {len(self.index)}")

    def _build_minute_ranges(self, message_path: Path) -> list[tuple[int, int, int]]:
        """Index the one-minute slices available in a single message file."""
        times = pd.read_parquet(message_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
        market_times = times[(times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS)]
        if market_times.size == 0:
            return []

        minute_ids = ((market_times - MARKET_OPEN_NS) // ONE_MINUTE_NS).astype(np.int64)
        minute_ids = np.clip(minute_ids, 0, None)
        unique_minutes = np.unique(minute_ids)
        if self.include_empty_minutes:
            minute_iterable = range(int(unique_minutes.min()), int(unique_minutes.max()) + 1)
        else:
            minute_iterable = unique_minutes.tolist()

        ranges: list[tuple[int, int, int]] = []
        for minute_id in minute_iterable:
            start = int(np.searchsorted(minute_ids, minute_id, side="left"))
            end = int(np.searchsorted(minute_ids, minute_id, side="right"))
            if self.include_empty_minutes or end > start:
                ranges.append((int(minute_id), start, end))

        return ranges

    def _load_features(self, file_idx: int) -> pd.DataFrame:
        """Load and cache per-order image features for one file."""
        if file_idx in self.cache:
            self.cache.move_to_end(file_idx)
            return self.cache[file_idx]

        msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
        snap_cols = ["Ask_Price_1", "Bid_Price_1"]
        messages = pd.read_parquet(self.message_files[file_idx], columns=msg_cols)
        snapshots = pd.read_parquet(self.snapshot_files[file_idx], columns=snap_cols)
        features = from_messages_and_snapshots_to_image_features(messages, snapshots, self.converters)

        self.cache[file_idx] = features
        self.cache.move_to_end(file_idx)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return features

    def __len__(self) -> int:
        """Return the number of indexed minute images."""
        return len(self.index)

    def __getitem__(self, idx: int):
        """Return one normalized HWC order image and lightweight metadata."""
        file_idx, minute_id, start, end = self.index[idx]
        features = self._load_features(file_idx)
        chunk = features.iloc[start:end]
        image = minute_chunk_to_order_image(chunk)
        image = image.astype(np.float32) / float(ORDER_IMAGE_MAX_VALUE)
        image = image * 2.0 - 1.0
        image = np.transpose(image, (1, 2, 0))
        return {
            "image": image,
            "file_idx": file_idx,
            "minute_id": minute_id,
        }


class OrderImageDataModule(pl.LightningDataModule):
    """Create the train and validation dataloaders for VQGAN training."""
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int,
        num_workers: int,
        converter_json_path: str,
        cache_size: int = 2,
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
        self.cache_size = int(cache_size)
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
            cache_size=self.cache_size,
            include_empty_minutes=self.include_empty_minutes,
            max_minutes_per_file=self.max_train_minutes_per_file,
            minute_stride=self.train_minute_stride,
        )
        self._val = RawMinuteOrderImageDataset(
            message_files=val_files,
            converter_json_path=self.converter_json_path,
            cache_size=self.cache_size,
            include_empty_minutes=self.include_empty_minutes,
            max_minutes_per_file=self.max_val_minutes_per_file,
            minute_stride=self.val_minute_stride,
        )

    def train_dataloader(self):
        """Build the shuffled training dataloader."""
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Build the validation dataloader."""
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


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


class ManualOptimizationVQWrapper(pl.LightningModule):
    """Adapt the CompVis VQ model to newer Lightning manual optimization."""
    def __init__(self, model: pl.LightningModule):
        super().__init__()
        self.model = model
        self.automatic_optimization = False
        self._loss_accepts_predicted_indices = "predicted_indices" in inspect.signature(self.model.loss.forward).parameters

    def forward(self, *args, **kwargs):
        """Forward to the wrapped VQ model."""
        return self.model(*args, **kwargs)

    def _call_loss(
        self,
        qloss,
        x,
        xrec,
        optimizer_idx: int,
        split: str,
        predicted_indices=None,
    ):
        """Call the loss with compatibility for different LDM signatures."""
        kwargs = {
            "last_layer": self.model.get_last_layer(),
            "split": split,
        }
        if self._loss_accepts_predicted_indices and predicted_indices is not None:
            kwargs["predicted_indices"] = predicted_indices
        return self.model.loss(
            qloss,
            x,
            xrec,
            optimizer_idx,
            self.global_step,
            **kwargs,
        )

    def _codebook_stats(self, indices):
        """Compute perplexity and active-code count from quantizer indices."""
        if isinstance(indices, (tuple, list)):
            indices = indices[0]
        flat = indices.reshape(-1).to(dtype=torch.long)
        counts = torch.bincount(flat, minlength=int(self.model.n_embed)).float()
        probs = counts / counts.sum().clamp_min(1.0)
        used = probs > 0
        perplexity = torch.exp(-(probs[used] * torch.log(probs[used] + 1e-10)).sum())
        cluster_usage = used.sum().to(dtype=torch.float32)
        return perplexity, cluster_usage

    def training_step(self, batch, batch_idx):
        """Run one manual-optimization training step with two optimizers."""
        opt_ae, opt_disc = self.optimizers()
        x = self.model.get_input(batch, self.model.image_key)
        xrec, qloss, ind = self.model(x, return_pred_indices=True)
        perplexity, cluster_usage = self._codebook_stats(ind)

        self.toggle_optimizer(opt_ae)
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self._call_loss(
            qloss,
            x,
            xrec,
            optimizer_idx=0,
            split="train",
            predicted_indices=ind,
        )
        self.manual_backward(aeloss)
        opt_ae.step()
        self.untoggle_optimizer(opt_ae)

        self.toggle_optimizer(opt_disc)
        opt_disc.zero_grad()
        discloss, log_dict_disc = self._call_loss(
            qloss,
            x,
            xrec,
            optimizer_idx=1,
            split="train",
        )
        self.manual_backward(discloss)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/perplexity", perplexity, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/cluster_usage", cluster_usage, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss.detach()

    def validation_step(self, batch, batch_idx):
        """Compute validation metrics for the wrapped VQ model."""
        x = self.model.get_input(batch, self.model.image_key)
        xrec, qloss, ind = self.model(x, return_pred_indices=True)
        perplexity, cluster_usage = self._codebook_stats(ind)

        aeloss, log_dict_ae = self._call_loss(
            qloss,
            x,
            xrec,
            optimizer_idx=0,
            split="val",
            predicted_indices=ind,
        )
        discloss, log_dict_disc = self._call_loss(
            qloss,
            x,
            xrec,
            optimizer_idx=1,
            split="val",
            predicted_indices=ind,
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/disc_loss", discloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/perplexity", perplexity, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/cluster_usage", cluster_usage, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        log_dict_ae = dict(log_dict_ae)
        log_dict_disc = dict(log_dict_disc)
        log_dict_ae.pop("val/rec_loss", None)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        if getattr(self.model, "use_ema", False):
            with self.model.ema_scope():
                xrec_ema, qloss_ema, ind_ema = self.model(x, return_pred_indices=True)
                aeloss_ema, log_dict_ae_ema = self._call_loss(
                    qloss_ema,
                    x,
                    xrec_ema,
                    optimizer_idx=0,
                    split="val_ema",
                    predicted_indices=ind_ema,
                )
                discloss_ema, log_dict_disc_ema = self._call_loss(
                    qloss_ema,
                    x,
                    xrec_ema,
                    optimizer_idx=1,
                    split="val_ema",
                    predicted_indices=ind_ema,
                )
                self.log("val_ema/rec_loss", log_dict_ae_ema["val_ema/rec_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_ema/aeloss", aeloss_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_ema/disc_loss", discloss_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
                log_dict_ae_ema = dict(log_dict_ae_ema)
                log_dict_disc_ema = dict(log_dict_disc_ema)
                log_dict_ae_ema.pop("val_ema/rec_loss", None)
                self.log_dict(log_dict_ae_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log_dict(log_dict_disc_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """Reuse the wrapped model optimizer configuration."""
        return self.model.configure_optimizers()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Forward the train-batch-end hook for EMA updates."""
        self.model.on_train_batch_end(outputs, batch, batch_idx)


def _instantiate_vq_model(config_path: Path, init_ckpt: str | None, learning_rate: float):
    """Build the VQ model and optionally load pretrained weights."""
    cfg = OmegaConf.load(str(config_path))
    cfg.model.params.ckpt_path = None
    model = instantiate_from_config(cfg.model)
    if init_ckpt:
        ckpt = torch.load(str(init_ckpt), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Restored from {init_ckpt} with {len(missing)} missing and {len(unexpected)} unexpected keys",
            flush=True,
        )
        if missing:
            print(f"Missing keys: {missing}", flush=True)
        if unexpected:
            print(f"Unexpected keys: {unexpected}", flush=True)
    model.learning_rate = float(learning_rate)
    return ManualOptimizationVQWrapper(model)


def main():
    """Parse arguments, build the trainer, and launch fitting."""
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=4.5e-6)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cache_size", type=int, default=2)
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
    p.add_argument("--converter_json_path", default=str(_find_default_converter_json()))
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
        cache_size=args.cache_size,
        include_empty_minutes=args.include_empty_minutes,
        max_train_minutes_per_file=args.max_train_minutes_per_file,
        max_val_minutes_per_file=args.max_val_minutes_per_file,
        train_minute_stride=args.train_minute_stride,
        val_minute_stride=args.val_minute_stride,
    )

    model = _instantiate_vq_model(
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
        filename="step={step}",
        monitor="val/rec_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
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

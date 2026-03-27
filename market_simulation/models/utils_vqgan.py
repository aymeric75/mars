from __future__ import annotations

import json
import inspect
import sys

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from market_simulation.models.utils import read_parquet_row_slice
from market_simulation.utils.bin_converter import BinConverter

REPO_ROOT = Path(__file__).resolve().parents[2]
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


@dataclass(frozen=True)
class ImageConverters:
    """Holds the bin converters used for order-image construction."""

    price_level: BinConverter
    order_volume: BinConverter


@lru_cache(maxsize=None)
def _build_converters_from_json(converter_json_path: str) -> ImageConverters:
    """Load image bin converters from a portable JSON file."""
    with Path(converter_json_path).open("r", encoding="utf-8") as f:
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


def create_mars_order_type_column(messages: pd.DataFrame) -> None:
    """Map raw message types and directions to the 3 MarS order classes.

    MarS_type meanings:
    0 = passive sell limits and aggressive buy limits
    1 = passive buy limits and aggressive sell limits
    2 = cancel/delete events
    """
    messages["Mars_type"] = 0
    messages.loc[messages["Message_Type"].isin([2, 3]), "Mars_type"] = 2
    messages.loc[
        ((messages["Message_Type"] == 1) & (messages["Direction"] == -1))
        | ((messages["Message_Type"] == 4) & (messages["Direction"] == 1)),
        "Mars_type",
    ] = 1


def create_subtle_order_type_column(messages: pd.DataFrame) -> None:
    """Map raw message types and directions to the 6 subtle order classes.

    Subtle_type meanings:
    0 = limit sell passive
    1 = limit buy passive
    2 = limit sell aggressive
    3 = limit buy aggressive
    4 = cancel/delete limit sell passive
    5 = cancel/delete limit buy passive
    """
    message_type = messages["Message_Type"].to_numpy(copy=False)
    direction = messages["Direction"].to_numpy(copy=False)
    subtle_type = np.full(len(messages), -1, dtype=np.int8)

    subtle_type[(message_type == 1) & (direction == 1)] = 0
    subtle_type[(message_type == 1) & (direction == -1)] = 1
    subtle_type[(message_type == 4) & (direction == 1)] = 2
    subtle_type[(message_type == 4) & (direction == -1)] = 3
    subtle_type[np.isin(message_type, [2, 3]) & (direction == 1)] = 4
    subtle_type[np.isin(message_type, [2, 3]) & (direction == -1)] = 5

    invalid_mask = subtle_type < 0
    if np.any(invalid_mask):
        invalid_rows = (
            messages.loc[invalid_mask, ["Message_Type", "Direction"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "Unsupported Message_Type/Direction combination for Subtle_type: "
            f"{invalid_rows}"
        )

    messages["Subtle_type"] = subtle_type


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

    return features[["Mars_type", "bin_price", "bin_vol"]]


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
    """Build one-minute order images on the fly from raw parquet row slices."""

    def __init__(
        self,
        message_files: list[str | Path],
        converter_json_path: str | Path,
        include_empty_minutes: bool = False,
        max_minutes_per_file: int | None = None,
        minute_stride: int = 1,
    ):
        self.include_empty_minutes = bool(include_empty_minutes)
        self.max_minutes_per_file = None if max_minutes_per_file is None else int(max_minutes_per_file)
        self.minute_stride = max(1, int(minute_stride))
        self.converters = _build_converters_from_json(str(Path(converter_json_path).resolve()))
        self.message_files: list[Path] = []
        self.snapshot_files: list[Path] = []
        self.index: list[tuple[int, int, int, int]] = []

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
            self.index.extend((file_idx, minute_id, raw_start, raw_rows) for minute_id, raw_start, raw_rows in minute_ranges)

        print(f"Total minute images: {len(self.index)}")

    def _build_minute_ranges(self, message_path: Path) -> list[tuple[int, int, int]]:
        """Index raw row slices corresponding to available one-minute samples."""
        times = pd.read_parquet(message_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
        market_mask = (times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS)
        market_rows = np.flatnonzero(market_mask)
        if market_rows.size == 0:
            return []

        market_times = times[market_rows]
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
            if end > start:
                raw_start = int(market_rows[start])
                raw_rows = int(market_rows[end - 1] - market_rows[start] + 1)
            elif self.include_empty_minutes:
                raw_start = int(market_rows[start]) if start < market_rows.size else int(market_rows[-1] + 1)
                raw_rows = 0
            else:
                continue
            ranges.append((int(minute_id), raw_start, raw_rows))

        return ranges

    def __len__(self) -> int:
        """Return the number of indexed minute images."""
        return len(self.index)

    def __getitem__(self, idx: int):
        """Return one normalized HWC order image and lightweight metadata."""
        file_idx, minute_id, raw_start, raw_rows = self.index[idx]
        msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
        snap_cols = ["Ask_Price_1", "Bid_Price_1"]
        messages = read_parquet_row_slice(
            self.message_files[file_idx],
            columns=msg_cols,
            start_row=raw_start,
            num_rows=raw_rows,
        )
        snapshots = read_parquet_row_slice(
            self.snapshot_files[file_idx],
            columns=snap_cols,
            start_row=raw_start,
            num_rows=raw_rows,
        )
        features = from_messages_and_snapshots_to_image_features(messages, snapshots, self.converters)
        image = minute_chunk_to_order_image(features)
        image = image.astype(np.float32) / float(ORDER_IMAGE_MAX_VALUE)
        image = image * 2.0 - 1.0
        image = np.transpose(image, (1, 2, 0))
        return {
            "image": image,
            "file_idx": file_idx,
            "minute_id": minute_id,
        }


class ManualOptimizationVQWrapper(pl.LightningModule):
    """Adapt the CompVis VQ model to newer Lightning manual optimization."""

    def __init__(self, model: pl.LightningModule):
        super().__init__()
        self.model = model
        self.automatic_optimization = False
        self._loss_accepts_predicted_indices = (
            "predicted_indices" in inspect.signature(self.model.loss.forward).parameters
        )

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


def instantiate_vq_model(config_path: Path | str, init_ckpt: str | None, learning_rate: float) -> ManualOptimizationVQWrapper:
    """Build the VQ model and optionally load pretrained weights."""
    cfg = OmegaConf.load(str(config_path))
    cfg.model.params.ckpt_path = None
    model = instantiate_from_config(cfg.model)
    if init_ckpt:
        ckpt = torch.load(str(init_ckpt), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        if any(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
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

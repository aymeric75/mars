import os, glob, zipfile, bisect
from functools import lru_cache
from typing import Optional, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler

import zarr
from zarr.storage import DirectoryStore

from utils import *  # build_model_from_variant, lm_loss_all_positions, etc.

import sys
sys.path.insert(0, "/projappl/project_2012747/mars/MarS")  # folder that contains market_simulation/
from market_simulation.models.order_model import OrderModel



# -------------------------
# DDP init
# -------------------------
def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def is_rank0():
    return rank() == 0


# -------------------------
# Validation
# -------------------------
@torch.no_grad()
def compute_val_loss(model, val_dl, device, use_amp: bool, amp_dtype, val_max_batches: Optional[int]):
    model.eval()
    total = 0.0
    count = 0

    for b, X in enumerate(val_dl, start=1):
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(X)
            loss = lm_loss_all_positions(logits, X)

        n = X.size(0) * (X.size(1) - 1)
        total += loss.item() * n
        count += n

        if val_max_batches is not None and b >= val_max_batches:
            break

    return total / max(1, count)


# -------------------------
# Data
# -------------------------
def unzip_zarr_zips(train_dir: str, pattern: str = "*.zarr.zip") -> List[str]:
    zips = sorted(glob.glob(os.path.join(train_dir, pattern)))
    out_dirs = []
    for zpath in zips:
        out_dir = zpath[:-4]  # strip ".zip" -> "... .zarr"
        out_dirs.append(out_dir)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
    return out_dirs


class MultiDirZarrOrderDataset(Dataset):
    def __init__(self, zarr_dirs: List[str], seq_len: int = 1024):
        self.seq_len = seq_len
        self.paths = list(zarr_dirs)

        self.lens = []
        for p in self.paths:
            X = zarr.open(DirectoryStore(p), path="X", mode="r")
            self.lens.append(max(0, X.shape[0] - seq_len - 1))

        self.cum = []
        s = 0
        for L in self.lens:
            s += L
            self.cum.append(s)

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    @staticmethod
    @lru_cache(maxsize=32)
    def _open_X(dir_path: str):
        store = DirectoryStore(dir_path)
        return zarr.open(store=store, path="X", mode="r")

    def __getitem__(self, idx: int):
        fi = bisect.bisect_right(self.cum, idx)
        prev = 0 if fi == 0 else self.cum[fi - 1]
        j = idx - prev

        X = self._open_X(self.paths[fi])
        x = X[j : j + self.seq_len]  # (1024, 15)
        return torch.from_numpy(x).long()


def make_train_val_loaders_ddp(
    ds,
    val_frac=0.2,
    seed=0,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    import random
    n = len(ds)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)

    train_sampler = DistributedSampler(train_set, shuffle=True, seed=seed, drop_last=drop_last)

    train_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )

    # simplest: only rank0 validates
    val_dl = None
    if is_rank0():
        val_dl = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(num_workers > 0),
        )

    return train_dl, val_dl, train_sampler


# -------------------------
# Train
# -------------------------
def train_fn():
    # -------------------------
    # Hyperparams
    # -------------------------
    target_global_batch = 4096
    seed = 42
    model_variant = "base"
    lr = 3e-4
    max_steps = 20000
    eval_every = 100
    val_max_batches = 200

    USE_AMP = True
    AMP_DTYPE = torch.bfloat16

    # per-GPU microbatch (what fits in memory)
    MICRO_BATCH = 32

    # IMPORTANT: global_batch = MICRO_BATCH * world_size * grad_accum
    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    ws = world_size()
    denom = MICRO_BATCH * ws
    if target_global_batch % denom != 0:
        raise ValueError(
            f"target_global_batch={target_global_batch} not divisible by MICRO_BATCH*world_size={denom}. "
            f"Pick a different MICRO_BATCH or global batch."
        )
    GRAD_ACCUM_STEPS = target_global_batch // denom

    if is_rank0():
        print(f"world_size={ws} | MICRO_BATCH(per GPU)={MICRO_BATCH} | GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} "
              f"| effective global batch={MICRO_BATCH * ws * GRAD_ACCUM_STEPS}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Data
    # -------------------------
    zarr_dirs = unzip_zarr_zips("../../data/order_model/train", "*_features.zarr.zip")
    ds = MultiDirZarrOrderDataset(zarr_dirs, seq_len=1024)

    # FIX: you previously passed train_fraction=0.8 as val_frac
    val_frac = 0.2

    train_dl, val_dl, train_sampler = make_train_val_loaders_ddp(
        ds,
        val_frac=val_frac,
        seed=seed,
        batch_size=MICRO_BATCH,   # per-GPU batch
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------
    # Model
    # -------------------------
    model, _ = build_model_from_variant(str(model_variant))
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    train_it = iter(train_dl)

    for step in range(1, max_steps + 1):
        train_sampler.set_epoch(step)

        opt.zero_grad(set_to_none=True)

        # gradient accumulation
        total_loss = 0.0
        for micro in range(GRAD_ACCUM_STEPS):
            try:
                X = next(train_it)
            except StopIteration:
                train_it = iter(train_dl)
                X = next(train_it)

            X = X.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                logits = model(X)
                loss = lm_loss_all_positions(logits, X)

            # scale loss so accumulated grads match big batch
            (loss / GRAD_ACCUM_STEPS).backward()
            total_loss += loss.item()

        opt.step()

        if is_rank0() and step % eval_every == 0 and val_dl is not None:
            # DDP wraps the real model in .module
            val_loss = compute_val_loss(model.module, val_dl, device, USE_AMP, AMP_DTYPE, val_max_batches)
            print(f"step {step} | train_loss {total_loss/GRAD_ACCUM_STEPS:.4f} | val_loss {val_loss:.4f}")

    ddp_cleanup()


if __name__ == "__main__":
    train_fn()

import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm


# ---------- constants ----------
NS_PER_MIN = 60 * 1_000_000_000
H = W = 32
C = 3
V_MAX = 256  # clip count to [0, 255] (uint8)



def decode_order_index_vec(idx_arr: np.ndarray) -> np.ndarray:
    idx = idx_arr.astype(np.int64, copy=False)
    idx //= 16
    volume_slot = idx % 32
    idx //= 32
    price_slot = idx % 32
    idx //= 32
    order_type = idx
    return np.stack([order_type, price_slot, volume_slot], axis=1)

def build_order_image(arr3: np.ndarray) -> np.ndarray:
    img = np.zeros((H, W, C), dtype=np.uint8)
    if arr3.size == 0:
        return img
    t = arr3[:, 0].astype(np.int64, copy=False)
    p = arr3[:, 1].astype(np.int64, copy=False)
    v = arr3[:, 2].astype(np.int64, copy=False)
    m = (t >= 0) & (t < C) & (p >= 0) & (p < W) & (v >= 0) & (v < H)
    t, p, v = t[m], p[m], v[m]
    key = (t * (H * W) + v * W + p)
    uniq, cnt = np.unique(key, return_counts=True)
    tt = uniq // (H * W)
    rem = uniq % (H * W)
    vv = rem // W
    pp = rem % W
    img[vv, pp, tt] = np.minimum(cnt, V_MAX).astype(np.uint8)
    return img



def parquet_to_memmap(
    input_dir,
    pattern="*.parquet",
    out_prefix="orders",
    time_col="Time",
    f0_col="f0",
    batch_size=500_000,   # bigger batch = fewer Python hops
):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise RuntimeError("No parquet files found")

    ds = load_dataset("parquet", data_files=[str(f) for f in files], split="train")
    n = len(ds)

    t_mm  = np.memmap(f"{out_prefix}_t.int64.mmap",  dtype=np.int64, mode="w+", shape=(n,))
    f0_mm = np.memmap(f"{out_prefix}_f0.int64.mmap", dtype=np.int64, mode="w+", shape=(n,))

    i = 0
    for batch in tqdm(ds.iter(batch_size=batch_size), total=(n // batch_size + 1)):
        bt  = np.asarray(batch[time_col], dtype=np.int64)
        bf0 = np.asarray(batch[f0_col],   dtype=np.int64)

        j = i + len(bt)
        t_mm[i:j]  = bt
        f0_mm[i:j] = bf0
        i = j

    t_mm.flush()
    f0_mm.flush()

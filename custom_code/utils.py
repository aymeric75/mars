import os
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



def decode_order_index_to_mmap(
    in_mmap: np.ndarray,
    out_path: str,
    chunk: int = 5_000_000,
    out_dtype=np.int32,   # int32 usually plenty for slots/types
):
    n = in_mmap.shape[0]

    out = np.memmap(out_path, mode="w+", dtype=out_dtype, shape=(n, 3))

    for s in range(0, n, chunk):
        e = min(s + chunk, n)

        # Read a chunk into RAM (small) so we can do in-place math safely
        tmp = np.array(in_mmap[s:e], dtype=np.int64, copy=True)

        tmp //= 16
        out[s:e, 2] = (tmp & 31).astype(out_dtype, copy=False)  # volume_slot

        tmp //= 32
        out[s:e, 1] = (tmp & 31).astype(out_dtype, copy=False)  # price_slot

        tmp //= 32
        out[s:e, 0] = tmp.astype(out_dtype, copy=False)         # order_type

    out.flush()
    return out



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


def parquet_to_memmap(f, time_col="Time", f0_col="f0", batch_size=500_000):


    f = Path(f)
    out = f.parent / f"{f.stem}_mmaps"
    out.mkdir(exist_ok=True)

    ds = load_dataset("parquet", data_files=str(f), split="train")
    n = len(ds)

    t = np.memmap(out / "t.int64.mmap",  "int64", "w+", shape=(n,))
    f0 = np.memmap(out / "f0.int64.mmap", "int64", "w+", shape=(n,))

    i = 0
    for b in tqdm(ds.iter(batch_size=batch_size), total=n // batch_size + 1):
        j = i + len(b[time_col])
        t[i:j]  = b[time_col]
        f0[i:j] = b[f0_col]
        i = j

    t.flush(); f0.flush()



# (n, 3)




WINDOW_NS = 60_000_000_000  # 60s in ns



_DTYPES = {
    "int32":  np.int32,
    "int64":  np.int64,
    "uint8":  np.uint8,
}

def read_mmap(path, cols=None):
    path = Path(path)
    dt = next(v for k, v in _DTYPES.items() if k in path.name)
    size = os.path.getsize(path) // np.dtype(dt).itemsize
    if cols:
        return np.memmap(path, dt, "r").reshape(size // cols, cols)
    return np.memmap(path, dt, "r")




def past_index_around_60s_ns(t_ns, seconds=60, none_value=None, return_object=True):
    """
    For each i, find the largest j <= i such that t_ns[i] - t_ns[j] >= seconds.
    If it doesn't exist, store `none_value` (default None).

    Parameters
    ----------
    t_ns : array-like (e.g., np.memmap), non-decreasing int64
    seconds : int or float
        Target lookback in seconds (default 60).
    none_value : any
        Value to use when no past index exists (default None).
        If you want an integer array, pass -1 (or another sentinel) and set return_object=False.
    return_object : bool
        If True, returns dtype=object with actual None for missing.
        If False, returns an integer array with `none_value` sentinel (must be int).

    Returns
    -------
    out : np.ndarray
        Array of past indices (same length as t_ns).
    """
    t_ns = np.asarray(t_ns)  # memmap stays zero-copy-ish for most ops
    if t_ns.ndim != 1:
        raise ValueError("t_ns must be a 1D array.")
    if not np.issubdtype(t_ns.dtype, np.integer):
        raise TypeError("t_ns must be an integer array (ns timestamps).")

    delta = np.int64(seconds * 1_000_000_000)  # seconds -> ns
    n = t_ns.shape[0]

    # For each i, we want earliest index k where t_ns[k] >= t_ns[i] - delta.
    # Then the answer is j = k (the first index at/after the cutoff),
    # BUT we need t_ns[j] <= t_ns[i] - delta (i.e., "at least delta ago"),
    # so actually we want the last index with t_ns <= cutoff, which is:
    # j = searchsorted(t_ns, cutoff, side="right") - 1
    cutoff = t_ns - delta
    j = np.searchsorted(t_ns, cutoff, side="right") - 1  # vectorized

    # j can be -1 when cutoff is before the first timestamp
    if return_object:
        out = j.astype(object)
        out[j < 0] = none_value  # typically None
        return out
    else:
        if none_value is None:
            raise ValueError("For return_object=False, none_value must be an integer sentinel (e.g., -1).")
        out = j.astype(np.int64)
        out[out < 0] = np.int64(none_value)
        return out

# Example usage:
# t = your np.memmap(...)
# idx_60s = past_index_around_60s_ns(t, seconds=60)          # dtype=object with None
# idx_60s_int = past_index_around_60s_ns(t, seconds=60, none_value=-1, return_object=False)  # int64 with -1





def build_image_mmap_from_lookback(
    idx_60s_int: np.ndarray,
    decoded: np.ndarray,              # can be np.memmap
    out_path: str,
    image_shape=(32, 32, 3),
    out_dtype=np.uint8,
):
    idx_60s_int = np.asarray(idx_60s_int)
    decoded = np.asarray(decoded)

    # if idx_60s_int.ndim != 1 or decoded.ndim != 1:
    #     raise ValueError("idx_60s_int and decoded must be 1D arrays.")
    if idx_60s_int.shape[0] != decoded.shape[0]:
        raise ValueError("idx_60s_int and decoded must have the same length.")
    if idx_60s_int.dtype.kind != "i":
        idx_60s_int = idx_60s_int.astype(np.int64, copy=False)

    N = idx_60s_int.shape[0]
    H, W, C = image_shape

    image_array = np.memmap(
        out_path,
        dtype=out_dtype,
        mode="w+",
        shape=(N, H, W, C),
    )
    image_array[:] = 0

    valid_i = np.flatnonzero(idx_60s_int != -1)

    for i in tqdm(valid_i, desc="Building order images", unit="img"):
        back = int(idx_60s_int[i])
        if back < 0 or back >= i:
            continue

        window = decoded[back:i]   # present index excluded

        img = build_order_image(window)

        img = np.asarray(img)
        if img.shape != (H, W, C):
            raise ValueError(
                f"build_order_image returned shape {img.shape}, expected {(H, W, C)}"
            )

        if img.dtype != out_dtype:
            img = img.astype(out_dtype, copy=False)

        image_array[i] = img

    image_array.flush()
    return image_array

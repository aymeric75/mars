import os
import pandas as pd
import numpy as np
import zarr
from zarr.storage import ZipStore
from zarr.storage import SQLiteStore
from typing import Union, Optional, Tuple
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from numcodecs import Blosc



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

    """
    Inputs:
        in_mmap: mmap file containing the order_index of all orders
    Output:
        creates a (N, 3) np mmap array giving for N order their type/price/volume
        type: is an integer in (0,1,2)
        price in [0, 32) indexes the slot of the price (previously discretized)
        volume in [0, 32) indexes the slot of the volume (previously discretized)
    """

    n = in_mmap.shape[0]

    out = np.memmap(out_path, mode="w+", dtype=out_dtype, shape=(n, 3))

    for s in range(0, n, chunk):

        e = min(s + chunk, n) # ending index of the current chunk

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
    """
    arr3: (N, 3) with columns [order_type, price_slot, volume_slot]
    returns: uint8 image (C, H, W)
      - C = order_type
      - H = volume_slot
      - W = price_slot
    """
    img = np.zeros((C, H, W), dtype=np.uint8)

    if arr3.size == 0:
        return img

    t = arr3[:, 0].astype(np.int64, copy=False)
    p = arr3[:, 1].astype(np.int64, copy=False)
    v = arr3[:, 2].astype(np.int64, copy=False)

    # keep only valid slots
    m = (t >= 0) & (t < C) & (p >= 0) & (p < W) & (v >= 0) & (v < H)
    t, p, v = t[m], p[m], v[m]

    key = (t * (H * W) + v * W + p)
    uniq, cnt = np.unique(key, return_counts=True)

    tt = uniq // (H * W)
    rem = uniq % (H * W)
    vv = rem // W
    pp = rem % W

    img[tt, vv, pp] = np.minimum(cnt, V_MAX).astype(np.uint8)
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
        if img.shape != (C, H, W):
            raise ValueError(
                f"build_order_image returned shape {img.shape}, expected {(C, H, W)}"
            )

        if img.dtype != out_dtype:
            img = img.astype(out_dtype, copy=False)

        image_array[i] = img

    image_array.flush()
    return image_array




def build_image_zarr_chunked_from_lookback(
    idx_60s_int: np.ndarray,
    decoded: np.ndarray,
    image_array,                 # a zarr array already created
    image_shape=(3, 32, 32),
    out_dtype=np.uint8,
):
    idx_60s_int = np.asarray(idx_60s_int)
    decoded = np.asarray(decoded)

    N = idx_60s_int.shape[0]
    C, H, W = image_shape

    # assume chunks like (chunkN, C, H, W)
    chunkN = image_array.chunks[0]

    # we'll iterate over chunk blocks [s, e)
    nblocks = (N + chunkN - 1) // chunkN

    for b in tqdm(range(nblocks), desc="Building order images (chunked)", unit="chunk"):
        s = b * chunkN
        e = min(N, s + chunkN)

        # build a full chunk buffer (default fill_value=0 semantics)
        buf = np.zeros((e - s, C, H, W), dtype=out_dtype)

        # indices within this block that are valid and in-range
        block_idx = np.arange(s, e, dtype=np.int64)
        mask = (idx_60s_int[s:e] != -1)
        valid_local = np.flatnonzero(mask)

        for j in valid_local:
            i = s + int(j)
            back = int(idx_60s_int[i])  # starting index for image of order at index i
            if back < 0 or back >= i:
                continue

            window = decoded[back:i]

            img = build_order_image(window)
            img = np.asarray(img)

            if img.shape != (C, H, W):
                raise ValueError(
                    f"build_order_image returned {img.shape}, expected {(C, H, W)}"
                )

            # HWC -> CHW
            #img = np.transpose(img, (2, 0, 1))

            if img.dtype != out_dtype:
                img = img.astype(out_dtype, copy=False)

            buf[j] = img



        # ONE write per block instead of (e-s) writes
        image_array[s:e] = buf






# def build_image_zipzarr_from_lookback(
#     idx_60s_int: np.ndarray,
#     decoded: np.ndarray,              # can still be np.memmap
#     out_path: str,                    # e.g. "images.zarr.zip" (one file)
#     image_shape=(32, 32, 3),
#     out_dtype=np.uint8,
#     chunks=None,                      # e.g. (256, 32, 32, 3)
#     compressor=None,                  # e.g. Blosc(...)
#     overwrite=True,
#     dataset_name="images",            # name inside the zip
# ):
#     idx_60s_int = np.asarray(idx_60s_int)
#     decoded = np.asarray(decoded)

#     if idx_60s_int.shape[0] != decoded.shape[0]:
#         raise ValueError("idx_60s_int and decoded must have the same length.")
#     if idx_60s_int.dtype.kind != "i":
#         idx_60s_int = idx_60s_int.astype(np.int64, copy=False)

#     N = idx_60s_int.shape[0]
#     H, W, C = image_shape

#     if chunks is None:
#         chunks = (min(1024, N), H, W, C)

#     if compressor is None:
#         compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

#     # One single file store (zip)
#     # mode:
#     # - "w" creates/overwrites the zip
#     # - "a" appends/updates existing zip
#     mode = "w" if overwrite else "a"

#     with ZipStore(out_path, mode=mode) as store:
#         # Create/open a group in Zarr v2 format inside the zip
#         root = zarr.group(store=store, overwrite=overwrite, zarr_format=2)

#         # Create the dataset; fill_value=0 avoids writing a full zero array up front
#         image_array = root.create_dataset(
#             name=dataset_name,
#             shape=(N, H, W, C),
#             chunks=chunks,
#             dtype=out_dtype,
#             compressor=compressor,
#             fill_value=0,
#             overwrite=overwrite,
#         )

#         valid_i = np.flatnonzero(idx_60s_int != -1)

#         for i in tqdm(valid_i, desc="Building order images", unit="img"):
#             back = int(idx_60s_int[i])
#             if back < 0 or back >= i:
#                 continue

#             window = decoded[back:i]   # present index excluded
#             img = build_order_image(window)

#             img = np.asarray(img)
#             if img.shape != (H, W, C):
#                 raise ValueError(
#                     f"build_order_image returned shape {img.shape}, expected {(H, W, C)}"
#                 )
#             if img.dtype != out_dtype:
#                 img = img.astype(out_dtype, copy=False)

#             image_array[i] = img  # persisted into the zip store

#         # Optional: ensure the zip central directory is updated now
#         store.flush()

#         return out_path, dataset_name













def from_mmap_to_zarr(
    input_mmap: str,
    mmap_shape: tuple[int, ...],
    mmap_type: Union[np.dtype, str],
    output_zarr: str,
    *,
    chunk_rows: int = 1024,
    clevel: int = 5,
    use_zipstore: bool = False,   # True => single .zip file, far fewer files
) -> None:
    dtype = np.dtype(mmap_type)

    # Memory-map the source (no big RAM use)
    mm = np.memmap(input_mmap, mode="r", dtype=dtype, shape=mmap_shape)

    # Choose store to control file explosion
    if use_zipstore:
        # output_zarr should end with .zip ideally
        store = ZipStore(output_zarr, mode="w")
    else:
        # output_zarr is a directory (many chunk files)
        store = output_zarr

    # Build chunks: chunk along axis 0, keep full remaining dims
    # Works for 1D, 2D, 4D images, etc.
    chunks = (min(chunk_rows, mmap_shape[0]),) + tuple(mmap_shape[1:])

    z = zarr.open(
        store,
        mode="w",
        shape=mmap_shape,
        chunks=chunks,
        dtype=dtype,
        compressor=Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE),
        zarr_format=2,
    )

    step = z.chunks[0]
    n = mmap_shape[0]

    for i in tqdm(range(0, n, step), desc="Converting mmap -> zarr", unit="chunk"):
        z[i:i + step] = mm[i:i + step]

    # Close ZipStore explicitly
    if use_zipstore:
        store.close()

import zarr
import numpy as np
import os
from zarr.storage import ZipStore
from pathlib import Path


with ZipStore("Vec_encoded_LOBSTER_Vec_encoded_LOBSTER_LOBSTER-AMZN-2025-12-11_order_images.zarr_tokens_v2.zarr.zip", mode="r") as store:
    arr = zarr.open(store=store, path="", mode="r")

    print(arr)

    #print(arr[:15].shape)
    # img = arr[123]        # single image
    # batch = arr[100:132] # batch


# exit()



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



# decoded = read_mmap("decoded.int32.mmap", cols=3)
# print(decoded.shape)


# t = read_mmap("t.int64.mmap")
# print(t.shape)
# print(t)

# #########

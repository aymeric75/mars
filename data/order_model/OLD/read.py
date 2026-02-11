import zarr
import numpy as np
import os
from zarr.storage import ZipStore
from pathlib import Path


with ZipStore("2025-12-09_features.zarr.zip", mode="r") as store:
    arr = zarr.open(store=store, path="X", mode="r")
    print(type(arr))
    print(arr.shape)
    print(arr[:15])
    #print(arr[:-15])



# window 1024
# batch size = 4096


# [
#     [41993     4     9    98 19799     0     2     0     0     0     0     0    0     0     3]
#     [12672     9     0    98 19799     0     2     0     0     0     0     0    0     0     3]
#     ....
# ]

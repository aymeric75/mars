import zarr
from zarr.storage import ZipStore
import os
import numpy as np

# convert zarr.zip to npz

# for fname in os.listdir("."):
#     if fname.endswith(".zarr.zip"):
#         base = fname.replace(".zarr.zip", "")
#         print(f"Converting {fname} -> {base}.npz")

#         store = ZipStore(fname, mode="r")
#         obj = zarr.open(store, mode="r")

#         np_arr = np.asarray(obj)

#         np.savez_compressed(base + ".npz", data=np_arr)

#         store.close()




# convert npz to zarr.zip in version 2 (must uninstall zarr v3 before and then install v2)

import os
import numpy as np
import zarr
from zarr.storage import ZipStore

for fname in os.listdir("."):
    if fname.endswith(".npz"):
        base = fname.replace(".npz", "")
        out_name = base + "_v2.zarr.zip"

        print(f"Converting {fname} -> {out_name}")

        # Load npz
        with np.load(fname) as npz:
            data = npz["data"]

        # Create Zarr v2 ZipStore
        store = ZipStore(out_name, mode="w")

        zarr.array(
            data,
            store=store,
            overwrite=True,
            zarr_version=2,   # 👈 force Zarr v2
        )

        store.close()









"""

import zarr
from numcodecs import Blosc
compressor = Blosc(
    cname="zstd",   # modern, very good compression
    clevel=5,       # 1–9, sweet spot around 3–5
    shuffle=Blosc.SHUFFLE,
)
store = zarr.ZipStore(NAME+"_v2.zarr.zip", mode="w")

z = zarr.create(
    shape=data.shape,
    dtype=data.dtype,
    chunks=(100_000, 64),   # tune this
    compressor=compressor,
    store=store,
    overwrite=True,
)

z[:] = data
store.close()



# # Group vs Array detection (works in v2 + v3)
# if hasattr(obj, "keys"):
#     print("This is a GROUP")
#     try:
#         print(obj.tree())
#     except AttributeError:
#         print("Group keys:", list(obj.keys()))
# elif hasattr(obj, "shape"):
#     print("This is an ARRAY")
#     print("shape:", obj.shape)
#     print("dtype:", obj.dtype)
#     print("chunks:", obj.chunks)
#     print("preview:", obj[:10] if obj.ndim == 1 else obj[0])
# else:
#     print("Unknown Zarr object")


# print(obj[:-150])


store.close()
"""

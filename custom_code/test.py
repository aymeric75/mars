from utils import *




# flat = read_mmap("../data/features/features_AAPL_2025-12-17_messages_10_mmaps/image_array.uint8.mmap")   # 1D memmap

# pix = 32*32*3

# N = flat.size // pix
# imgs = flat.reshape(N, 32, 32, 3)

# print(imgs.shape)


# from_mmap_to_zarr(
#     "../data/features/features_AAPL_2025-12-17_messages_10_mmaps/image_array.uint8.mmap",
#     (5227808, 32, 32, 3),
#     np.uint8,
#     "../data/features/features_AAPL_2025-12-17_messages_10_mmaps/image_array.zarr",
#     use_zipstore = True
# )


# import zarr
# from zarr.storage import ZipStore

path = "../data/features/features_AAPL_2025-12-17_messages_10_mmaps/image_array.zarr"
store = ZipStore(path, mode="r")

# v2 read:
arr = zarr.open_array(store, mode="r", zarr_format=2)

# print(arr.shape, arr.dtype)
# store.close()


# print(type(arr))


# import numpy as np
# from tqdm import tqdm

# mm = np.memmap("../data/features/features_AAPL_2025-12-17_messages_10_mmaps/image_array.uint8.mmap", mode="r", dtype=np.uint8)

# k = 20   # how many values to print
# found = []

# chunk_size = 1_000_000

# for i in range(0, mm.size, chunk_size):
#     chunk = mm[i:i + chunk_size]
#     nz = np.nonzero(chunk)[0]

#     if nz.size > 0:
#         for j in nz[:k - len(found)]:
#             found.append((i + j, chunk[j]))
#         if len(found) >= k:
#             break

# for idx, val in found:
#     print(f"index={idx}, value={val}")

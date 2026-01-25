from utils import *

""" from features .parquets files into Time/f0 mmap files (loaded as 'numpy.memmap' ), then into order images files (mmap) and finally  into discrete tokens (VQGAN learned representation) """




base_dir = Path("../data/features")

exclude = {
    "features_all.parquet",
}



# 1) creating empty dirs
for f in base_dir.glob("features_*.parquet"):
    if f.name not in exclude:
        (base_dir / f"{f.stem}_mmaps").mkdir(exist_ok=True)




# 2) From features parquets files create f0 and t (Time) .mmap
# for f in base_dir.glob("features_*.parquet"):
#     if f.name not in exclude:
#         parquet_to_memmap(f)



# 3) From Time/f0 mmap files create decoded (N, 3) ARRAY
# for f in base_dir.glob("features_*.parquet"):
#     if f.name in exclude:
#         continue

#     mmaps = f.parent / f"{f.stem}_mmaps"
#     in_mmap = np.memmap(mmaps / "f0.int64.mmap", dtype=np.int64, mode="r")

#     decode_order_index_to_mmap(
#         in_mmap,
#         mmaps / "decoded.int32.mmap",
#     )


# 4) FROM decoded AND time arrays create mmap IMAGE ORDER

for f in base_dir.glob("features_*.parquet"):
    if f.name in exclude:
        continue

    mmaps = f.parent / f"{f.stem}_mmaps"

    decoded = read_mmap( mmaps / "decoded.int32.mmap", cols=3)
    t = read_mmap( mmaps / "t.int64.mmap", cols=None)

    print(decoded)

    print(t)

    image_mmap = build_image_mmap_from_lookback(
        idx_60s_int=idx_60s_int,
        decoded=decoded,                 # or f0 if already int64 array/memmap
        out_path= Path (mmaps  / "image_array.mmap"),
        out_dtype=np.uint8,                # adjust as needed
    )



    print()


# #idx_60s = past_index_around_60s_ns(t, seconds=60)          # dtype=object with None
# idx_60s_int = past_index_around_60s_ns(t, seconds=60, none_value=-1, return_object=False)

# # print(len(idx_60s)) # 5227808


# image_mmap = build_image_mmap_from_lookback(
#     idx_60s_int=idx_60s_int,
#     decoded=decoded,                 # or f0 if already int64 array/memmap
#     out_path="image_array.mmap",
#     out_dtype=np.uint8,                # adjust as needed
# )


# go over idx_60s_int (of length N) and whenever it is not -1, take the corresponding value (of the "back index until which we can look at") and present index
#
#  and create take from  f0.int64 (also of length N) , all the values froom back index to present index (excluded)
#          pass all these values to build_order_image , then fill in a (N, 32, 32, 3) array (well... you should create an empty zeros one at the begining), THEN
#
#

# une fois que tu as le putain

# import zarr
# z = zarr.open(
#     "image_array.zarr",
#     mode="w",
#     shape=(N, 32, 32, 3),
#     chunks=(1024, 32, 32, 3),
#     dtype=np.uint8,
#     compressor=zarr.Blosc(cname="zstd", clevel=5)
# )





# [34200001448678 34200001525151 34200005742081 ... 57599994287924 57599994287924 57599997733830]

# # loop all chosen parquets (same exclude list as before)
# for f in base_dir.glob("*.parquet"):
#     if f.name in exclude:
#         continue
#     file_to_images(f)


# order_images = np.memmap(base_dir / "features_AAPL_2025-12-17_messages_10_mmaps" / "order_images.uint8.mmap", mode="r")


# imgs = np.memmap(
#     base_dir / "features_AAPL_2025-12-17_messages_10_mmaps" / "order_images.uint8.mmap",
#     dtype=np.uint8,
#     mode="r",
#     shape=(3072, 32, 32, 3),   # N must match what you created
# )

# print(imgs.shape)


#order_idx = np.memmap(base_dir / "features_AAPL_2025-12-17_messages_10_mmaps" / "order_idx.int32.mmap" mode="r")

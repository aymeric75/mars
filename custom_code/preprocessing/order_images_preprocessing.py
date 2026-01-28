import zipfile
import sys
from pathlib import Path

custom_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(custom_folder))


from utils import *

""" from features .parquets files into Time/f0 mmap files (loaded as 'numpy.memmap' ), then into order images files (mmap) and finally  into discrete tokens (VQGAN learned representation) """




base_dir = Path("../../data/features")

exclude = {
    "features_all.parquet",
}



# 1) creating empty dirs
for f in base_dir.glob("features_*.parquet"):
    if f.name not in exclude:
        (base_dir / f"{f.stem}_mmaps").mkdir(exist_ok=True)



# 1bis) create a zarr.zip file for each feature_*parquet
for f in base_dir.glob("features_*.parquet"):
    if f.name not in exclude:
        mmaps = f.parent / f"{f.stem}_mmaps"
        date_str = str(f.name.split("_")[2])
        parquet_to_zarr_zip(
            f,
            mmaps / (date_str + "_features.zarr.zip"),
        )

exit()

# 2) From features parquets files create f0 and t (Time) .mmap
# for f in base_dir.glob("features_*.parquet"):
#     if f.name not in exclude:
#         parquet_to_memmap(f)


# 3) From Time/f0 mmap files create decoded (N, 3) array (the 3 dims are for type/price/volume)
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

include = {
    #"features_NVDA_2025-12-22_messages_10.parquet",
    #"features_AMZN_2025-12-11_messages_10.parquet",
    "features_NVDA_2025-12-22_messages_10.parquet",
    #"features_TSLA_2025-12-17_messages_10.parquet",
    #"features_TSLA_2025-12-12_messages_10.parquet"
}

for f in base_dir.glob("features_*.parquet"):

    if f.name in exclude:
        continue
    # if f.name not in include:
    #     continue

    mmaps = f.parent / f"{f.stem}_mmaps"

    decoded = read_mmap( mmaps / "decoded.int32.mmap", cols=3)
    t = read_mmap( mmaps / "t.int64.mmap", cols=None)


    idx_60s_int = past_index_around_60s_ns(t, seconds=60, none_value=-1, return_object=False)

    # print(type(idx_60s_int))
    # print(len(idx_60s_int[idx_60s_int == -1]))
    # print(len(idx_60s_int[idx_60s_int != -1]))


    idx = np.where(idx_60s_int != -1)[0][0]


    date_str = str(f.name.split("_")[2])  # '2025-12-09'
    #date = datetime.strptime(date_str, "%Y-%m-%d").date()

    H = 32
    W = 32
    C = 3


    save_images_tail_zip(
        Path(mmaps  / (date_str+"_order_images.zarr.zip")),
        Path(mmaps  / (date_str+"_order_images_pruned.zarr.zip")),
        start_idx=idx,
    )


    continue

    with ZipStore(Path(mmaps  / "order_images.zarr.zip"), mode="w", compression=zipfile.ZIP_DEFLATED) as store:
        root = zarr.group(store=store, overwrite=True)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        arr = root.create_dataset(
            "images",
            shape=(len(decoded), C, H, W),
            chunks=(1024, C, H, W),
            dtype="u1",
            compressor=compressor,  # <-- turn on chunk compression
            fill_value=0,
            overwrite=True,
        )


        build_image_zarr_chunked_from_lookback(idx_60s_int, decoded, arr, image_shape=(C,H,W))
        store.flush()


    # image_mmap = build_image_mmap_from_lookback(
    #     idx_60s_int=idx_60s_int,
    #     decoded=decoded,                 # or f0 if already int64 array/memmap
    #     out_path= Path (mmaps  / "image_array.uint8.mmap"),
    #     out_dtype=np.uint8,                # adjust as needed
    # )














    print("image_mmap")



# #idx_60s = past_index_around_60s_ns(t, seconds=60)          # dtype=object with None
# idx_60s_int = past_index_around_60s_ns(t, seconds=60, none_value=-1, return_object=False)

# # print(len(idx_60s)) # 5227808


# image_mmap = build_image_mmap_from_lookback(
#     idx_60s_int=idx_60s_int,
#     decoded=decoded,                 # or f0 if already int64 array/memmap
#     out_path="image_array.mmap",
#     out_dtype=np.uint8,                # adjust as needed
# )


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

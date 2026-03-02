import os
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import zarr
from numcodecs import Blosc
from zarr.storage import ZipStore

from utils_preproc import decode_order_index_df, past_index_around_60s_ns, build_image_zarr_chunked_from_lookback


""" from features .parquets files into *order-images.zarr.zip files holding the order images """


print("HELLO")

#


# dir with all "*_features.parquet" files
INPUT_DIR = Path("/scratch/project_2012747/mars_data/order_model/train/final")
OUTPUT_DIR = Path("/scratch/project_2012747/mars_data/order_batch_model/train/raw")



EXCLUDE = {
    "features_all.parquet",
}

INCLUDE = {
    # e.g. "features_NVDA_2025-12-22_messages_10.parquet",
}


def should_process(f: Path) -> bool:
    if f.name in EXCLUDE:
        return False
    if INCLUDE and f.name not in INCLUDE:
        return False
    return True


def process_one_file(f_str: str, input_dir_str: str, output_dir_str: str) -> str:
    f = Path(f_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    # Load parquet
    df = pd.read_parquet(f)

    # Decode f0 -> order_type/price_slot/volume_slot
    decoded = decode_order_index_df(df)

    # Compute lookback indices (~60s)
    t = df["Time"]
    idx_60s_int = past_index_around_60s_ns(t, seconds=60, none_value=-1, return_object=False)

    # Find first valid idx (same logic as original)
    first_valid = np.where(idx_60s_int != -1)[0]
    if len(first_valid) == 0:
        return f"SKIP (no valid 60s lookback): {f.name}"



    order_images_file_name = f.stem.replace("features", "order_images") + ".zarr.zip"
    final_path = output_dir / order_images_file_name
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    # Skip if already fully completed
    if final_path.exists():
        return f"SKIP (exists): {final_path.name}"

    # Clean up leftover tmp from crashed runs
    if tmp_path.exists():
        tmp_path.unlink()



    # Create zarr.zip and write images
    H = W = 32
    C = 3

    with ZipStore(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as store:
        root = zarr.group(store=store, overwrite=True)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        arr = root.create_dataset(
            "images",
            shape=(len(decoded), C, H, W),
            chunks=(1024, C, H, W),
            dtype="u1",
            compressor=compressor,
            fill_value=0,
            overwrite=True,
        )

        build_image_zarr_chunked_from_lookback(idx_60s_int, decoded, arr, image_shape=(C, H, W))
        store.flush()

    # Atomic rename to final name (safe commit)
    tmp_path.rename(final_path)

    return f"OK: {final_path.name}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*_features.parquet"))
    files = [f for f in files if should_process(f)]


    if not files:
        print("No files to process.")
        return

    # Choose worker count (can also set explicitly)
    max_workers = min(os.cpu_count() or 1, len(files))
    print(f"Processing {len(files)} files with {max_workers} processes...")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for f in files:
            futures.append(ex.submit(process_one_file, str(f), str(INPUT_DIR), str(OUTPUT_DIR)))

        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    main()




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

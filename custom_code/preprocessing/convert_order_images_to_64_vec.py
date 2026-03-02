# Standard library
import os
import sys
import math
import re
import shutil
import zipfile
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pytorch_lightning as pl
import zarr
from zarr.storage import ZipStore
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse

# PyTorch utilities
from torch.utils.data import Dataset, DataLoader
print(os.cpu_count())
# -------------------------
# Add external repos
# -------------------------
BASE = Path("/projappl/project_2012747/mars_derrick_branch/third_party")
CKPT_DIR = Path("/scratch/project_2012747/Mars_Derrick/checkpoints/checkpoint_downsample_100")
ZIP_DIR = Path("/scratch/project_2012747/mars_data/order_batch_model/train/raw/")
ZIP_DIR_PROCESSED = Path("/scratch/project_2012747/mars_data/order_batch_model/train/final/")
OUT_DIR = Path("/scratch/project_2012747/mars_data/order_batch_model/train/final/")

sys.path.insert(0, str(BASE / "latent_diffusion"))
sys.path.insert(0, str(BASE / "taming-transformers"))

OUT_DIR.mkdir(exist_ok=True)

# -------------------------
# Lightning wrapper
# -------------------------
class VQForOrders(pl.LightningModule):
    def __init__(self, vqmodel):
        super().__init__()
        self.m = vqmodel

    def forward(self, x):
        q, _, _ = self.m.encode(x)
        return self.m.decode(q)

# -------------------------
# Load best checkpoint
# -------------------------
def find_best_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    def extract_val_loss(p):
        m = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", p.name)
        return float(m.group(1))
    return min(ckpts, key=extract_val_loss)

def load_model(best_ckpt):
    from ldm.util import instantiate_from_config
    cfg = OmegaConf.load(BASE / "latent_diffusion/models/first_stage_models/vq-f4/config.yaml")
    vq = instantiate_from_config(cfg.model)
    model = VQForOrders.load_from_checkpoint(
        checkpoint_path=best_ckpt,
        vqmodel=vq,
        strict=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

# -------------------------


class ZarrDataset(Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]
        return torch.from_numpy(x).float().div_(255.0)

from numcodecs import Blosc
import threading

def process_zip_file(zip_path: Path, model, device, batch_size=1024):
    print(f"Processing {zip_path.name}")

    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    # Fast unzip
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    zarr_folders = [p for p in extract_dir.iterdir() if p.suffix == ".zarr"]
    zarr_folder = zarr_folders[0] if zarr_folders else extract_dir

    root = zarr.open(zarr_folder, mode="r")
    arr = root["images"]

    num_samples = arr.shape[0]

    # -------------------------
    # Determine token dimension once
    # -------------------------
    encode = model.m.encode

    with torch.inference_mode():
        sample = torch.zeros((1, *arr.shape[1:]), device=device)
        _, _, info = encode(sample)

        tokens = info[2]
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]

        token_dim = tokens.numel()
        
    stem = get_filename(zip_path)
    
    out_zarr = OUT_DIR / f"{stem}_64vectors.zarr"

    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    out = zarr.open(
        str(out_zarr),
        mode="w",
        shape=(num_samples, token_dim),
        chunks=(batch_size, token_dim),
        dtype=np.float32,
        compressor=Blosc(
            cname="lz4",
            clevel=1,
            shuffle=Blosc.BITSHUFFLE
        ),
        zarr_format=2
    )

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    stream = torch.cuda.Stream()

    write_idx = 0
    pending_write = None

    def async_write(start, data):
        out[start:start + data.shape[0]] = data

    # -------------------------
    # Double buffer setup
    # -------------------------
    next_batch_np = arr[0:batch_size]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):

        for i in tqdm(
            range(0, num_samples, batch_size),
            desc=zip_path.name,
            smoothing=0,
            mininterval=1,
        ):

            batch_np = next_batch_np

            # Preload next batch early (IO overlap)
            next_i = i + batch_size
            if next_i < num_samples:
                next_batch_np = arr[next_i:next_i + batch_size]

            # Pinned memory staging
            batch = torch.from_numpy(batch_np).pin_memory()

            with torch.cuda.stream(stream):
                x = batch.to(
                    device,
                    non_blocking=True,
                    dtype=torch.float16
                ).div_(255)

                _, _, info = encode(x)

                tokens = info[2]
                if isinstance(tokens, (tuple, list)):
                    tokens = tokens[0]

                tokens = tokens.view(x.size(0), -1).cpu().numpy()

            torch.cuda.current_stream().wait_stream(stream)

            # Async disk write
            if pending_write is not None:
                pending_write.join()

            pending_write = threading.Thread(
                target=async_write,
                args=(write_idx, tokens),
            )
            pending_write.start()

            write_idx += tokens.shape[0]

        if pending_write:
            pending_write.join()

    shutil.make_archive(str(out_zarr), "zip", root_dir=out_zarr)
    shutil.rmtree(extract_dir)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Process only zarr files starting with this symbol (e.g. GOOG)"
    )
    # ignore unknown args (needed in Jupyter)
    args, _ = parser.parse_known_args()
    return args
    
def get_filename(zip_path: Path, file_path_filter= "_order_images"):
    stem = zip_path.name
    stem = stem.replace(f"{file_path_filter}.zarr.zip", "")
    stem = stem.replace(f"{file_path_filter}.zarr", "")
    stem = stem.replace(".zarr.zip", "")   # remove leftover .zarr.zip
    stem = stem.replace(".zarr", "")       # remove leftover .zarr
    return stem
    

def main():
    args = parse_args()
    symbol = args.symbol
    best_ckpt = find_best_checkpoint(CKPT_DIR)
    model, device = load_model(best_ckpt)
    '''

    # Read once (DO NOT overwrite later)
    all_zip_files = list(ZIP_DIR.glob("*.zarr.zip"))

    # Read all zip files
    #all_zip_files = list(ZIP_DIR.glob("*.zarr.zip"))

    raw_symbol_files = [
        p for p in all_zip_files
        if p.name.startswith(symbol_arg)
    ]

    print(f"Total raw files: {len(raw_symbol_files)}")

    for zip_file in raw_symbol_files:
        print(f"Processing: {zip_file}")
        process_zip_file(zip_file, model, device)



    '''
    
    files_to_process = []


    all_zip_files = list(ZIP_DIR.glob("*.zarr.zip"))
    all_processed_files = list(ZIP_DIR_PROCESSED.glob("*.zarr.zip"))

    # Filter per symbol
    raw_symbol_files = [
        p for p in all_zip_files
        if p.name.startswith(symbol)
    ]

    processed_symbol_files = [
        p for p in all_processed_files
        if p.name.startswith(symbol)
    ]

    # Extract base names
    raw_names = {
        get_filename(p, "_order_images"): p
        for p in raw_symbol_files
    }

    processed_names = {
        get_filename(p, "_64vectors")
        for p in processed_symbol_files
    }

    # Find unprocessed
    missing = [
        raw_names[name]
        for name in raw_names
        if name not in processed_names
    ]

    print(f"{symbol}: {len(missing)} files to process")

    files_to_process.extend(missing)

    print("\nFinal files to process:")
    for f in files_to_process:
        print(f)


    for zip_file in files_to_process:
        process_zip_file(zip_file, model, device)
        

        
if __name__ == "__main__":
    main()

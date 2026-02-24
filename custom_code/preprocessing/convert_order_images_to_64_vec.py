# process to parquet 
import os
import sys
from pathlib import Path
import re
import zipfile
import zarr
import shutil
import torch
from zarr.storage import ZipStore
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------
# Add external repos
# -------------------------
BASE = Path("/projappl/project_2012747/mars_derrick_branch/third_party")
CKPT_DIR = Path("/scratch/project_2012747/Mars_Derrick/checkpoints/checkpoint_downsample_100")
ZIP_DIR = Path("/scratch/project_2012747/mars_data/order_batch_model/train/intermediate/")
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
# Fast Zarr dataset
# -------------------------
class ZarrDataset(Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]
        return torch.from_numpy(x).float().div_(255.0)

# -------------------------
# Processing function
# -------------------------
def process_zip_file(zip_path: Path, model, device, batch_size=512):
    print(f"Processing {zip_path.name} ...")

    # extract to temp folder
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # the extracted folder should be a .zarr folder
    zarr_folders = [p for p in extract_dir.iterdir() if p.is_dir() and p.suffix == ".zarr"]
    if not zarr_folders:
        # fallback: maybe the folder itself is the zarr
        zarr_folder = extract_dir
    else:
        zarr_folder = zarr_folders[0]

    print(f"Using extracted folder: {zarr_folder}")

    # open Zarr
    root = zarr.open(zarr_folder, mode="r")
    arr = root["images"]
    num_samples = arr.shape[0]

    all_tokens = []

    dataset = ZarrDataset(arr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    with torch.inference_mode():
        for i, x in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda"):
                _, _, info = model.m.encode(x)
            B = x.size(0)
            tokens = info[2].view(B, -1).cpu().numpy()
            all_tokens.append(tokens)

            if i % 20 == 0:
                print(f"{zip_path.name}: processed {min((i+1)*batch_size, num_samples)}/{num_samples}")

    all_tokens = np.vstack(all_tokens)

    # save as .zarr folder
    out_zarr_folder = OUT_DIR / f"{zip_path.stem.replace('_order_images','')}_64_vector.zarr"
    if out_zarr_folder.exists():
        shutil.rmtree(out_zarr_folder)
    out_root = zarr.open(str(out_zarr_folder), mode="w", shape=all_tokens.shape,
                         chunks=(batch_size, all_tokens.shape[1]), dtype=all_tokens.dtype)
    out_root[:] = all_tokens

    # zip the output
    shutil.make_archive(str(out_zarr_folder), 'zip', root_dir=out_zarr_folder)
    print(f"Saved zipped Zarr: {str(out_zarr_folder)}.zip")

    # cleanup
    shutil.rmtree(extract_dir)

# -------------------------
# Main
# -------------------------
def main():
    best_ckpt = find_best_checkpoint(CKPT_DIR)
    model, device = load_model(best_ckpt)

    zip_files = list(ZIP_DIR.glob("*.zarr.zip"))
    print(f"Found {len(zip_files)} zip files to process.")

    for zip_file in zip_files:
        process_zip_file(zip_file, model, device)

if __name__ == "__main__":
    main()

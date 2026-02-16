import torch
import zarr
import numpy as np
import sys
import os
import re
from pathlib import Path
from zarr.storage import ZipStore
from torch.utils.data import Dataset
from omegaconf import OmegaConf
import bisect
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from zarr.storage import ZipStore
from torchvision.utils import save_image



class VQForOrders(pl.LightningModule):
    """ VQGAN model class, adapted for Orders """
    def __init__(self, vqmodel, lr=1e-4):
        super().__init__()
        self.m = vqmodel
        self.lr = lr

    def forward(self, x):
        q, _, info = self.m.encode(x)
        
        return self.m.decode(q)

    def training_step(self, batch, _):
        x = batch
        x_rec = self(x)
        loss = ((x_rec - x) ** 2).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x = batch
        x_rec = self(x)
        val_loss = ((x_rec - x) ** 2).mean()
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



def load_best_model(best_ckpt_path, repo_root):
    from ldm.util import instantiate_from_config

    cfg = OmegaConf.load(
        repo_root / "Mars_Derrick/third_party/latent-diffusion/models/first_stage_models/vq-f4/config.yaml"
    )

    vq = instantiate_from_config(cfg.model)

    model = VQForOrders.load_from_checkpoint(
        checkpoint_path=best_ckpt_path,
        vqmodel=vq,
        lr=1e-4,
        strict=True
    )
    

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    model.to(device)
    model.eval()

    return model, device

def find_best_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")

    def extract_val_loss(p):
        m = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", p.name)
        if m is None:
            raise ValueError(f"Could not parse val_loss from filename: {p.name}")
        return float(m.group(1))

    best_ckpt = min(ckpts, key=extract_val_loss)
    return best_ckpt
    

class FolderZarrDataset(Dataset):
    def __init__(self, path):


        self.arrays = []
        self.lengths = []

        
        zarr_name = path.name.replace(".zip", "")
        dataset_name = f"{zarr_name}/images"

        store = ZipStore(str(path), mode="r")
        arr = zarr.open(store=store, path=dataset_name, mode="r")

        print(f"Loaded {path} with shape {arr.shape}")

   

        self.arrays.append(arr)
        self.lengths.append(arr.shape[0])

        # cumulative lengths for indexing
        self.cum_lengths = []
        total = 0
        for l in self.lengths:
            total += l
            self.cum_lengths.append(total)

    def __len__(self):
        return self.cum_lengths[-1]

BATCH_SIZE = 2048  # adjust according to GPU memory
device = torch.device("cuda")

dataset_path = Path("/scratch/project_2012747/Data_zarr/Training_VQGAN/LOBSTER-TSLA-2025-12-22_order_images.zarr.zip")
OUT_DIR = Path("./latents_zip")
OUT_DIR.mkdir(exist_ok=True)

THIS_DIR = Path(os.getcwd()).resolve().parent
REPO = THIS_DIR.parents[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(REPO/ "Mars_Derrick" / "third_party" / "latent-diffusion"))
sys.path.insert(0, str(REPO/ "Mars_Derrick"  / "third_party" / "taming-transformers"))

#Finding best model 
BEST_CKPT_PATH = Path("/scratch") / "project_2012747" / "Mars_Derrick" / "checkpoints" / "checkpoint_downsample_100/" 
BEST_CKPT = find_best_checkpoint(BEST_CKPT_PATH)


# ---- Load model ----
model, device = load_best_model(BEST_CKPT, REPO)
Best_model = model

model = model.to(device)
model.eval()

dataset = FolderZarrDataset(dataset_path)

for file_idx, arr in enumerate(dataset.arrays):
    num_samples = arr.shape[0]
    original_file_name = dataset_path.stem
    out_file = OUT_DIR / f"{original_file_name}_tokens.zarr.zip"

    # Zarr store
    store = ZipStore(str(out_file), mode="w")
    tokens_arr = zarr.zeros(
        shape=(num_samples, 64),
        chunks=(BATCH_SIZE, 64),
        dtype="i4",
        store=store,
        overwrite=True
    )

    next_report = 50_000

    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)
        batch_imgs = arr[start:end]

        # Convert to tensor once, normalize in one step
        x_input = torch.as_tensor(batch_imgs, dtype=torch.float32, device=device)
        x_input = x_input / 255.0  # simplest way


    

        with torch.inference_mode():
            _, _, info = model.m.encode(x_input)

            B = x_input.size(0)
            indices = info[2].view(B, 8, 8)
            tokens = indices.reshape(B, -1)  # flatten to 64

        # move to CPU once
        tokens_arr[start:end] = tokens.cpu().numpy()

        while end >= next_report:
            print(f"Processed {next_report}/{num_samples}")
            next_report += 50_000

    store.close()
    print(f"Saved token Zarr: {out_file} with shape {tokens_arr.shape}")

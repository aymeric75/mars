import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from torchvision.utils import save_image

#
# ---------- Dataset ----------
class OrderArray(Dataset):
    def __init__(self, x):  # x: (N,3,32,32)
        assert x.ndim == 4 and x.shape[1:] == (3, 32, 32)
        x = x.astype(np.float32)

        mx = float(x.max())
        if mx > 1.0:
            x /= (100.0 if mx <= 100.0 else 255.0)  # [0,1]
        x = x * 2.0 - 1.0  # [-1,1]

        self.x = torch.from_numpy(x)

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i]



class VQForOrders(pl.LightningModule):
    def __init__(self, vqmodel, lr=1e-4):
        super().__init__()
        self.m = vqmodel
        self.lr = lr

    def forward(self, x):
        q, _, _ = self.m.encode(x)
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



# ---------- Train ----------
def train():


    REPO = Path.cwd().resolve().parent

    # access the VQGAN official code
    sys.path.insert(0, str(REPO / "third_party" / "latent-diffusion"))
    sys.path.insert(0, str(REPO / "third_party" / "taming-transformers"))

    from ldm.util import instantiate_from_config


    # Load the order images (here in a npy file)
    data = np.load(REPO / "data" / "order_images.npy")  # (4142,3,32,32) # 9 days worth data

    # create the dataloader (for train and val)
    ds = OrderArray(data)
    n = len(ds)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                            generator=torch.Generator().manual_seed(0))

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)


    # load VQGan model with config and weights as in MarS paper
    cfg = OmegaConf.load(str(REPO / "third_party/latent-diffusion/models/first_stage_models/vq-f4/config.yaml"))
    vq = instantiate_from_config(cfg.model)

    ckpt = torch.load(REPO / "custom_code" / "model.ckpt", map_location="cpu", weights_only=False)
    vq.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    vq.learning_rate = 1e-4
    vq.train()

    model = VQForOrders(vq, lr=1e-4)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_loss:.6f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=20,
        check_val_every_n_epoch=2,   # <-- val every X epochs
        log_every_n_steps=50,
        callbacks=[ckpt_cb],
    )

    trainer.fit(model, train_dl, val_dl)
    print("Best checkpoint:", ckpt_cb.best_model_path)

    # Post training, test one sample using BEST weights + save gt/pred
    best = VQForOrders.load_from_checkpoint(ckpt_cb.best_model_path, vqmodel=vq, lr=1e-4)
    best.eval()
    best.to("cuda" if torch.cuda.is_available() else "cpu")

    x = test_ds[0].unsqueeze(0).to(best.device)  # (1,3,32,32)
    with torch.no_grad():
        x_rec = best(x)

    # Save as images (map [-1,1] -> [0,1])
    gt = (x[0].cpu() + 1) / 2
    pr = (x_rec[0].cpu() + 1) / 2
    save_image(gt, REPO / "custom_code" / "gt.png")
    save_image(pr, REPO / "custom_code" / "pred.png")
    print("Saved:", REPO / "custom_code" / "gt.png", "and", REPO / "custom_code" / "pred.png")

train()

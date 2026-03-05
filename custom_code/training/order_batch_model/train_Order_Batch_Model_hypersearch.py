# train_Order_Batch_Model_lightning.py
import argparse
import glob
import os
import torch
import lightning.pytorch as pl

from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

from market_simulation.models.order_batch_model import OrderBatchModel
from market_simulation.models.utils_order_batch_model import TokenDataset, lm_loss_next_token




class OrderBatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        pattern: str,
        array_path: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pattern = pattern
        self.array_path = array_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self._train = None
        self._val = None

    def setup(self, stage: str | None = None):
        # use extracted dirs only: *.zarr.zip -> *.zarr
        self._train = TokenDataset(sorted(Path(self.train_dir).glob("*.npy")))
        self._val = TokenDataset(sorted(Path(self.val_dir).glob("*.npy")))
        
        
    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )





class OrderBatchLightningModule(pl.LightningModule):
    def __init__(self, emb_dim: int, num_layers: int, num_heads: int, vocab_size: int, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.model = OrderBatchModel(
            emb_dim=int(emb_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            vocab_size=int(vocab_size),
        )
        self.lr = float(lr)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)  # (B,T,V)

    def training_step(self, batch, batch_idx):
        input_ids = batch  # (B,1024)
        logits = self(input_ids)
        loss = lm_loss_next_token(logits, input_ids)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        logits = self(input_ids)
        loss = lm_loss_next_token(logits, input_ids)
        self.log("val_loss", loss, on_step=False, sync_dist=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)





class PrintCkptFilename(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                fp = cb.format_checkpoint_name(
                    metrics=trainer.callback_metrics,
                    filename=cb.filename,
                )
                # fp may already be absolute depending on Lightning version/config
                full = fp if os.path.isabs(fp) else os.path.join(cb.dirpath, fp)
                print(">> checkpoint dirpath:", cb.dirpath)
                print(">> checkpoint full path:", full)



def main():
    p = argparse.ArgumentParser()
    # Hypersearch-style output control (like order_model hypersearch)
    p.add_argument("--run_root", default="checkpoints_batch_order_model")
    p.add_argument("--run_name", default=None)
    
    p.add_argument("--train_dir", default="/scratch/project_2012747/mars_data/order_batch_model/train/final")
    p.add_argument("--val_dir", type=str, default="/scratch/project_2012747/mars_data/order_batch_model/val/final", required=True)
    p.add_argument("--pattern", default="*.npy")
    p.add_argument("--array_path", default="")
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--emb_dim", type=int, default=128) #default=768)
    p.add_argument("--num_layers", type=int, default=4) #default=12)
    p.add_argument("--num_heads", type=int, default=4) #default=12)
    p.add_argument("--vocab_size", type=int, default=8192)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--default_root_dir", default="")
    args = p.parse_args()
    
    
    run_root = args.run_root
    os.makedirs(run_root, exist_ok=True)
    
    # default run_name if not provided (keep it short but informative)
    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"
    
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print("PWD =", os.getcwd())
    print("run_root =", os.path.abspath(run_root))
    print("run_name =", run_name)
    print("run_dir  =", os.path.abspath(run_dir))


    pl.seed_everything(args.seed, workers=True)

    dm = OrderBatchDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        pattern=args.pattern,
        array_path=args.array_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = OrderBatchLightningModule(
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        lr=args.lr,
    )
    
    logger = TensorBoardLogger(
        save_dir=run_root,
        name="tensorboard",
        version=run_name,   # distinct TB run per hyperparam combo
    )
    
    
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[ckpt_cb, PrintCkptFilename()],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.is_available() else "auto",
        max_steps=args.max_steps,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=50,
        limit_val_batches=10,
        enable_checkpointing=True,
        enable_progress_bar=False,
        deterministic=True,
    )
    


    trainer.fit(model, dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

import argparse
import os
import torch
import lightning.pytorch as pl

from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from market_simulation.models.order_batch_model import OrderBatchModel
from market_simulation.models.utils_order_batch_model import (
    OnlineMessageTokenDataset,
    VQRuntimeConfig,
    lm_loss_next_token,
)


class OrderBatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        pattern: str,
        batch_size: int,
        num_workers: int,
        cache_size: int,
        vq_runtime: VQRuntimeConfig,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pattern = pattern
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_size = int(cache_size)
        self.vq_runtime = vq_runtime
        self._train = None
        self._val = None

    def setup(self, stage: str | None = None):
        train_files = sorted(Path(self.train_dir).glob(self.pattern))
        val_files = sorted(Path(self.val_dir).glob(self.pattern))
        self._train = OnlineMessageTokenDataset(
            message_files=[str(p) for p in train_files],
            cache_size=self.cache_size,
            vq_runtime=self.vq_runtime,
        )
        self._val = OnlineMessageTokenDataset(
            message_files=[str(p) for p in val_files],
            cache_size=self.cache_size,
            vq_runtime=self.vq_runtime,
        )

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
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch
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
                fp = cb.format_checkpoint_name(metrics=trainer.callback_metrics, filename=cb.filename)
                full = fp if os.path.isabs(fp) else os.path.join(cb.dirpath, fp)
                print(" >> checkpoint dirpath:", cb.dirpath)
                print(" >> checkpoint full path:", full)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", default="checkpoints_batch_order_model")
    p.add_argument("--run_name", default=None)
    p.add_argument("--train_dir", default="/scratch/project_2012747/mars_data/train")
    p.add_argument("--val_dir", default="/scratch/project_2012747/mars_data/val")
    p.add_argument("--pattern", default="*messages*.parquet")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cache_size", type=int, default=2)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--latent_diffusion_root", default="../../../third_party/latent_diffusion")
    p.add_argument("--taming_root", default="../../../third_party/taming-transformers")
    p.add_argument("--vq_ckpt_dir", default="./")
    args = p.parse_args()

    run_root = args.run_root
    os.makedirs(run_root, exist_ok=True)
    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("PWD =", os.getcwd())
    print("run_root =", os.path.abspath(run_root))
    print("run_name =", run_name)
    print("run_dir =", os.path.abspath(run_dir))

    pl.seed_everything(args.seed, workers=True)

    vq_runtime = VQRuntimeConfig(
        ckpt_dir=args.vq_ckpt_dir,
        latent_diffusion_root=args.latent_diffusion_root,
        taming_root=args.taming_root,
    )

    dm = OrderBatchDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        pattern=args.pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        vq_runtime=vq_runtime,
    )

    model = OrderBatchLightningModule(
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        lr=args.lr,
    )

    logger = TensorBoardLogger(save_dir=run_root, name="tensorboard", version=run_name)
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

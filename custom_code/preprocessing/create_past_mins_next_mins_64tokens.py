from pathlib import Path
import re
import numpy as np
import pandas as pd
import zarr


def build_context_zarrs(
    zarr_dir: str,
    indices_dir: str,
    output_dir: str,
):
    zarr_dir = Path(zarr_dir)
    indices_dir = Path(indices_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # images_tokens_AMZN-2025-12-11.zarr
    pat = re.compile(r"images-tokens_(?P<stock>[^-]+)-(?P<date>\d{4}-\d{2}-\d{2})\.zarr")

    for zarr_path in zarr_dir.glob("*.zarr"):
        m = pat.match(zarr_path.name)
        if not m:
            continue

        print("OK1")
        stock = m.group("stock")
        date = m.group("date")

        past_path = indices_dir / f"past16_{stock}_{date}_cut.parquet"
        next_path = indices_dir / f"next1_{stock}_{date}_cut.parquet"

        if not past_path.exists() or not next_path.exists():
            continue

        print("OK2")
        past_df = pd.read_parquet(past_path)
        next_df = pd.read_parquet(next_path)

        # load image tokens (N, 64)
        tokens = zarr.open(zarr_path, mode="r")

        
        print("OK3")

        # indices
        past_idx = past_df[[k for k in range(16)]].to_numpy(dtype=np.int64)   # (M, 16)
        next_idx = next_df[0].to_numpy(dtype=np.int64)                       # (M,)
        
        # gather all rows we need in ONE zarr read
        flat = np.concatenate([past_idx.reshape(-1), next_idx])
        uniq, inv = np.unique(flat, return_inverse=True)  # uniq is sorted; inv maps flat -> uniq positions
        
        tok_uniq = tokens[uniq]  # ONE big read: (U, 64)
        
        # rebuild past and next from the unique token table
        M = past_idx.shape[0]
        past_tokens = tok_uniq[inv[: M * 16]].reshape(M, 16, 64)
        next_tokens = tok_uniq[inv[M * 16 :]]            # (M, 64)

        
        """
        # build past (num_samples, 16, 64)
        past_tokens = np.stack(
            [
                tokens[past_df[k].to_numpy()]
                for k in range(16)
            ],
            axis=1,
        )
        

        # build next (num_samples, 64)
        next_tokens = tokens[next_df[0].to_numpy()]
        """
        print("OK4")
        
        # save as zarr.zip (v2)
        zarr.save(
            output_dir / f"past16_tokens_{stock}-{date}.zarr.zip",
            past_tokens,
        )
        zarr.save(
            output_dir / f"next1_tokens_{stock}-{date}.zarr.zip",
            next_tokens,
        )


build_context_zarrs(
    "../../data/order_batch_model",
    "../../data/order_batch_model/results",
    "../../data/order_batch_model",
)

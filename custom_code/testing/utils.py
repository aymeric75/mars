import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import random
import torch

from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState
from market_simulation.utils.bin_converter import BinConverter
from market_simulation.models.ensemble_model import EnsembleModel
from market_simulation.models.utils_order_model import build_model_from_variant
from market_simulation.models.order_batch_model import OrderBatchModel
from market_simulation.models.order_model import OrderModel









# Load the Order Model
def load_order_model(ckpt_path: str, device: str = "cpu", K: int = 1024):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})

    model, _ = build_model_from_variant(model_variant=hp["model_variant"], K=K)

    state_dict = ckpt["state_dict"]
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model



# Load the Order Batch Model
def load_order_batch_model(ckpt_path: str, device: str = "cpu"):

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # (A) recover model hyperparams saved by `self.save_hyperparameters()`
    hp = ckpt.get("hyper_parameters", {})
    model = OrderBatchModel(
        emb_dim=int(hp["emb_dim"]),
        num_layers=int(hp["num_layers"]),
        num_heads=int(hp["num_heads"]),
        vocab_size=int(hp.get("vocab_size", 8192)),
    )

    # (B) load weights: Lightning stored them under "model.*"
    sd = ckpt["state_dict"]
    sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)

    model.eval()

    return model


 # Load the Ensemble Model (PT file)
def load_ensemble_model(ckpt_path: str, device: str = "cpu", order_vocab_size: int = 49152):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # this is the dict saved by save_checkpoint()
    state_dict = ckpt["model_state_dict"]             # <-- critical: NOT "state_dict"

    model = EnsembleModel(order_vocab_size=int(order_vocab_size))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


"""
# Load the Ensemble Model
def load_ensemble_model(ckpt_path: str, order_vocab_size: int, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = EnsembleModel(order_vocab_size=order_vocab_size)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.to(device)
    model.eval()
    return model
"""

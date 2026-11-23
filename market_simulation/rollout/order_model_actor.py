from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import ray

from market_simulation.conf import C
from custom_code.testing.utils import load_order_model


@ray.remote(num_gpus=1)
class OrderModelActor:
    def __init__(self):
        self.model = load_order_model(
            ckpt_path="step=step=3360-val=val_loss=3.7445.ckpt",
            device="cuda",
        ).eval()
        if C.model_serving.fp16:
            self.model.half()

        self.temperature = C.model_serving.temperature

    def predict(self, arr: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        x = torch.from_numpy(arr.copy()).cuda()
        x = x.reshape((1, C.order_model.seq_len, C.order_model.token_dim))

        with torch.no_grad():
            out = self.model.sample(x, self.temperature).int().cpu().numpy().reshape(-1)
            # out = self.model.sample(x, self.temperature).int().cpu().reshape((1, -1)).numpy()

        # return out[0]
        return out.astype(np.int32)

from __future__ import annotations

import numpy as np
import ray
import os


class RayModelClient:
    def __init__(self, actor_name: str = "order_model_actor"):
        if not ray.is_initialized():
            ray.init(
                address=os.environ["MARS_RAY_ADDRESS"],
                namespace="mars",
                ignore_reinit_error=True,
            )
        self.actor = ray.get_actor(actor_name, namespace="mars")

    def get_prediction(self, state_vector):
        return ray.get(self.actor.predict.remote(state_vector))

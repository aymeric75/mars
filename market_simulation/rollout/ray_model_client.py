from __future__ import annotations

import numpy as np
import ray

class RayModelClient:
    def __init__(self, actor_name: str = "order_model_actor"):
        self.actor = ray.get_actor(actor_name)

    def get_prediction(self, state_vector: np.ndarray) -> np.ndarray:
        return ray.get(self.actor.predict.remote(state_vector))

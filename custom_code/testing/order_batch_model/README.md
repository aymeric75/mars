`order_batch_model_speed.py`

- Loads the Order-Batch model and the VQ model once.
- Precomputes the old 16-minute feature history once.
- Times preprocessing only: last raw row -> last feature row -> append/prune rolling 16-minute feature buffer -> 16 order images -> `16 x 64` VQ tokens.
- Times inference only: one next-token prediction from the `1024`-token prefix.

Results

- Current one-token benchmark: `preprocess_ms=97.056`, `one_token_inference_ms=4.686`, `total_ms=101.742`.
- Previous full-rollout benchmark before the one-token change: `preprocess_ms=86.847`, `inference_ms=357.881`, `total_ms=444.729`.

`order_batch_model_speed.py`

- Loads the Order-Batch model and the VQ model once.
- Precomputes the old 16-minute feature history once.
- Times preprocessing only: last raw row -> last feature row -> rolling 16-minute feature buffer -> 16 order images -> `16 x 64` VQ tokens.
- Times inference only: one next-token prediction from the `1024`-token prefix.

Results

- `device=cuda`
- `preprocess_ms=97.056`, `one_token_inference_ms=4.686`, `total_ms=101.742`

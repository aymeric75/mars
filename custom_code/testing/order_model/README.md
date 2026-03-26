`order_model_speed.py`

- Loads the checkpoint once.
- Loads one raw market-hours slice once.
- Times preprocessing only: raw rows -> feature batch tensor.
- Times inference only: feature batch -> `order_model(x)`.

Results

- `seq_len=1024`, `batch_size=16`, `device=cuda`
- `preprocess_ms=8.878`, `inference_ms=31.829`, `total_ms=40.708`

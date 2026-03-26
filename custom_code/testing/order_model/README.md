`order_model_speed.py`

- Loads the checkpoint once.
- Loads one raw market-hours slice once.
- Times preprocessing only: raw rows -> feature batch tensor.
- Times inference only: feature batch -> `order_model(x)`.

Results

- Observed in discussion: inference is under `100 ms`.

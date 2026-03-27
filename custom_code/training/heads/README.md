# Probability Head Training

The probability head was started with:

```bash
HEAD_TYPE=probability
SCENARIO=order_model
TRADE_SIDE=long
TRAIN_SAMPLES_PER_EPOCH=3072
VAL_SAMPLES=1024
MAX_STEPS=20000
PNL_MARGIN=1000
```

`PNL_MARGIN=1000` is used to ignore near-zero raw PnL examples when training the probability head. Concretely, samples with `abs(pnl) <= 1000` are discarded for the probability loss, so the model focuses on clearer positive/negative cases.

Even with this margin filter, the results are still not good.

![Probability head validation loss](./figures/val_loss_proba_head_30s.png)

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

For the previous 2-class probability head, a validation loss around `0.6` meant the binary cross-entropy was below the naive `~0.69` baseline, so the model was learning something, but the result was still not especially strong.

![Probability head validation loss](./figures/val_loss_proba_head_30s.png)

For the newer 3-class probability head, the validation loss looks more stable:

![3-class probability head validation loss](./figures/val_proba_loss_30sec_3classes.png)

The validation accuracy is also fairly high, which suggests the 3-class formulation is easier to learn than the previous binary setup:

![3-class probability head validation accuracy](./figures/val_proba_acc_30sec_3classes.png)

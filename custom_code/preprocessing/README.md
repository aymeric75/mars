# Order Model data preprocessing

Using `test_cpus.slurm` call messages_to_features.py, uncomment / comment each step in the main function (I. to III.)



# Order Batch Model data preprocessing

## create the order images from the features.parquet files

`order_images_preprocessing.py` creates for each `features.parquet`, a `zarr.zip` file of the same length. The first images should only zeros. The `idx_60s_int` variable should be (?) saved somewhere since it holds the indices of where the right images start.

The `.zarr.zip` created can be used to train the VQGAN.



## create "-16 min" indices and "+1 min" index for each order

`create_past_mins_next_mins_indices.py`: the `create_min16_plus1_indices_from_feature_parquets` function goes over a folder of `*features.parquet` files and for each creates:

- `past16*.parquet` (for each order, hold 16 indices (<=> last 16 minutes))
- `next1*.parquet` (same as past16 but holds the next one minute index)
- `features_cut.parquet`  (input feature file but without the orders (at the begining and at the end) that do not have -16 or +1 min worth of data)


`create_past_mins_next_mins_64tokens.py`: go over a folder of 64-tokens zarr.zip file, and create two other "64-tokens" zarr.zip, one with the -16 (min) 64-tokens, one with +1 (min) 64-token.


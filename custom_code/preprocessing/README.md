# Order Model data preprocessing

Using `test_cpus.slurm` call messages_to_features.py, uncomment / comment each step in the main function (I. to III.)



# Order Batch Model data preprocessing

## create the order images from the features.parquet files

`order_images_preprocessing.py` creates for each `features.parquet`, a `zarr.zip` file of the same length. The first images should only zeros. The `idx_60s_int` variable should be (?) saved somewhere since it holds the indices of where the right images start.



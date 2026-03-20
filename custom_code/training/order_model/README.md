# Training 

stocks: AAPL, AMD, AVGO, META, NFLX

Training set: 3-11-2025 -> 10-11-2025

Eval set: 17-11-2025 -> 19-11-2025


## Train data sampling

Each training sample is a sliding window of length `K` taken from one `*_messages.parquet` file after filtering to market hours only (`09:30` to `16:00`).

For one file with `N` valid rows, the dataset creates:

`N - K + 1`

possible windows, with stride `1`.

So if `K = 1024`, sample `i` from a file is:

- rows `[i : i + 1024]`

This means consecutive samples overlap heavily.

During training, the code does not iterate through all windows in plain file order. Instead, it uses a custom sampler:

1. For each file, all its windows are split into chunks of size `train_chunk_size` (default: `2048` windows).
2. All chunks from all files are mixed together.
3. The sampler shuffles the order of the chunks.
4. Inside each selected chunk, it shuffles the windows again.
5. It keeps yielding windows until `train_num_samples` windows have been produced.

So the training order is:

- random across chunks
- random inside each chunk
- without replacement inside one sampler pass
- limited to `train_num_samples` if this argument is set


## Eval data sampling

Validation samples are built from the validation `*_messages.parquet` files in the same way as training samples:

- each sample is a sliding window of length `K`
- for one file with `N` valid rows, this gives `N - K + 1` possible windows
- sample `i` corresponds to rows `[i : i + K]`

Unlike training, validation does not use the custom chunk-based sampler.

Instead, the validation `DataLoader` reads the validation dataset directly with a standard batching scheme:

- by default, `shuffle_val=False`
- so validation windows are read in dataset order
- that means the loader goes through the validation files in sorted order, and inside each file it goes through window starts `0, 1, 2, ...`

So validation sampling is deterministic by default:

- no chunking
- no random reordering
- no partial sampling limit like `train_num_samples`

If `shuffle_val=True` is enabled, then the validation windows are shuffled at the `DataLoader` level before batching, but this is disabled by default.

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq


def read_parquet_row_slice(
    parquet_path,
    columns,
    start_row=None,
    num_rows=None,
    batch_size=65536,
):
    """Read only a row slice from a parquet file while selecting columns."""
    if start_row is None:
        return pd.read_parquet(parquet_path, columns=columns)

    if num_rows is None or num_rows <= 0:
        return pd.DataFrame(columns=columns)

    stop_row = start_row + num_rows
    parquet_file = pq.ParquetFile(parquet_path)
    batches = []
    seen_rows = 0

    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        batch_len = len(batch)
        batch_start = seen_rows
        batch_stop = seen_rows + batch_len

        if batch_stop <= start_row:
            seen_rows = batch_stop
            continue

        if batch_start >= stop_row:
            break

        take_start = max(start_row, batch_start) - batch_start
        take_stop = min(stop_row, batch_stop) - batch_start
        batches.append(batch.slice(take_start, take_stop - take_start))
        seen_rows = batch_stop

    if not batches:
        return pd.DataFrame(columns=columns)

    return pd.concat([batch.to_pandas() for batch in batches], ignore_index=True)

import os
import re

# Directory containing your files
directory = "META"

# Prefix you want to add
symbol = "META"

# Regex pattern to match your files
pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(messages|snapshots|meta)_\d+\.parquet")

for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        date, filetype = match.groups()

        new_name = f"{symbol}_{date}_{filetype}.parquet"

        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        print(f"Renaming: {filename} -> {new_name}")
        os.rename(old_path, new_path)

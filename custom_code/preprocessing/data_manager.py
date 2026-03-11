import lobdatamanager as ldm
from datetime import date
from collections import defaultdict
from pathlib import Path

base_dir =  Path("/scratch/project_2012747/mars_data") #Path().resolve().parent.parent
dest_folder = Path(base_dir / "order_model/val/raw") #Path(base_dir / "order_model/test")

pattern = "_snapshots"

date1 = date(2025, 11, 28)
date2 =  date(2025, 12, 3)


def order_by_date(items):
    """
    Return a dict mapping each input string to its chronological order
    based on the YYYY-MM-DD date after the '/' separator.
    """
    dates = {x: date.fromisoformat(x.split("/")[1]) for x in items}
    sorted_items = sorted(items, key=lambda x: dates[x])
    return {x: i for i, x in enumerate(sorted_items)}

def group_by_date(items):
    """
    Return a SORTED (oldest dates first) dict mapping each date to the list of strings
    having that date (YYYY-MM-DD after '/').
    """
    grouped = defaultdict(list)
    for x in items:
        grouped[date.fromisoformat(x.split("/")[1])].append(x)
    return dict(sorted(grouped.items()))

def cut_by_date(d, start, end):
    """
    Return a dict with items whose date key is in [start, end].
    """
    return {k: v for k, v in d.items() if start <= k <= end}


# 2025-11-01


#-------------------------------------------------------------------------------------
# Retrieve all day/stocks available, for a given period and store them in all_pairs
#-------------------------------------------------------------------------------------
allas_storage = ldm.AllasStorage(config_file=".csc_creds.json", timeout=100, retries=3)
files = ldm.list_allas_files(
    project_name="project_2012747",
    bucket_name="NewNasdaq",
    folder="LOBSTER",
    storage_instance=allas_storage
)
only_messages = [x for x in files if pattern+"_10.parquet" in x]
dico = {}
for f in only_messages:
    key_ = f.replace("LOBSTER/", "").replace(pattern+"_10.parquet", "")
    dico[key_] = f
list_stock_date = list(dico.keys())
ordered_list_stock_date = order_by_date(list_stock_date)
# like:
# {
# 'MSFT/2025-12-15': 0,
# 'TSLA/2026-01-07': 1,
# 'AAPL/2026-02-01': 2
# }
day_pairs = group_by_date(ordered_list_stock_date)

# like:
# {datetime.date(2025, 10, 1): ['AAPL/2025-10-01', 'AMD/2025-10-01', ...],
# datetime.date(2025, 10, 2): ['AA
result = cut_by_date(
    day_pairs,
    start=date1,
    end=date2,
)
all_pairs = []
for k, v in result.items():
    all_pairs.extend(v)


#--------------------------------------------------------
# Retrieve data of the days/stock and put them somewhere
#--------------------------------------------------------
for pair in all_pairs:
    # LOBSTER/AAPL/2026-01-12_messages_10.parquet
    df = ldm.read_allas_file(
        project_name="project_2012747",
        bucket_name="NewNasdaq",
        file_name=dico[pair],
        storage_instance=allas_storage
    )

    pair = pair.replace("/", "_")

    #print(dest_folder / str(pair + "_messages" + ".parquet"))

    df.to_parquet(dest_folder / str(pair + pattern + ".parquet"))


    #allas_storage.download("kanniain", "A7_LOBSTER", "local_file.txt", "distant_file.txt")

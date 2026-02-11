import lobdatamanager as ldm
from datetime import date
from collections import defaultdict

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

outfolder =

date1 = date(2026, 11, 1)

date2 =  date(2026, 11, 26)




#--------------------------------------------------------
# Retrieve all day/stocks available
#--------------------------------------------------------

allas_storage = ldm.AllasStorage(config_file=".csc_creds.json")

files = ldm.list_allas_files(
    project_name="project_2012747",
    bucket_name="NewNasdaq",
    folder="LOBSTER/",
    storage_instance=allas_storage
)

#print(files)



#--------------------------------------------------------
# Retrieve all day/stocks available for a given period
#--------------------------------------------------------

#print(order_by_date(files))

only_messages = [x for x in files if "_messages_10.parquet" in x]
dico = {}
for f in only_messages:
    key_ = f.replace("LOBSTER/", "").replace("_messages_10.parquet", "")
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
    start=date(2026, 1, 10),
    end=date(2026, 1, 31),
)

print(result)
all_pairs = []

for k, v in result.items():
    all_pairs.extend(v)

#print(all_pairs)

# print(dico["AAPL/2026-01-12"]) # LOBSTER/AAPL/2026-01-12_messages_10.parquet


# Given 2 dates,    dl dans un endroit donné

import lobdatamanager as ldm

allas_storage = ldm.AllasStorage(".csc_creds.json", timeout=100, retries=3)

# # list buckets for a specific project from allas_storage
# allas_storage.list_buckets("project_2012747")

# retrieve the S3 ressource corresponding to the project
s3_resource = allas_storage.get_s3_resource("kanniain")


# 2. List files in a specific folder
files = ldm.list_allas_files(project_name="kanniain", bucket_name="A7Data", folder="raw/", storage_instance=allas_storage)

from pathlib import Path

from tqdm import tqdm

for f in tqdm(files):
    if Path(f).exists():
        continue

    Path(f).parent.mkdir(parents=True, exist_ok=True)

    s3_resource.Bucket("A7Data").download_file(Filename=f, Key=f)


# # uploading a file (key is the distant file)
# s3_resource.Bucket("NewNasdaq").upload_file(
#     Filename="local_file.txt",
#     Key="distant_file.txt"
# )
# #

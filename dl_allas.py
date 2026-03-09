import lobdatamanager as ldm

allas_storage = ldm.AllasStorage(".csc_creds.json", timeout=100, retries=3)

# # list buckets for a specific project from allas_storage
# allas_storage.list_buckets("project_2012747")

# retrieve the S3 ressource corresponding to the project
s3_resource = allas_storage.get_s3_resource("project_2012747")



# 2. List files in a specific folder
files = ldm.list_allas_files(
    project_name="project_2012747",
    bucket_name="NewNasdaq",
    folder="LOBSTER/META/",
    storage_instance=allas_storage
)

print(files)

for f in files:

    s3_resource.Bucket("NewNasdaq").download_file(
        Filename="META/"+f.split("/")[-1],
        Key=f
    )

# # uploading a file (key is the distant file)
# s3_resource.Bucket("NewNasdaq").upload_file(
#     Filename="local_file.txt",
#     Key="distant_file.txt"
# )
# #

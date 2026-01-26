import lobdatamanager as ldm

"""

To initialize an Allas storage, call the AllasStorage class which takes either of:


No argument but retrieve the csc username and password from environement variables:
export ALLAS_USERNAME=YOUR_CSC_ID
export ALLAS_USERNAME=YOUR_CSC_PASSWORD

or a .csc_creds.json file, like:

{
    "id": CSC_USERNAME,
    "password": CSC_PASSWORD
}

or a .boto3_credentials file, like:

[s3allas-project_2015660]
type = s3
provider = Other
env_auth = false
aws_access_key_id = some_access_key_id
aws_secret_access_key = some_secret_access
endpoint = a3s.fi
acl = private

[s3allas-project_2015683]
etc.


The .boto3_credentials can file can be created following these steps:
1) connect to puthi/mahti web interface and go to Cloud Storage configuration
2) create Allas S3 credentials for the chosen projects ("remotes")
3) go to the Login node shell and execute: sed -E 's/^(access|secret)/aws_\1/g' ~/.config/rclone/rclone.conf > ~/.boto3_credentials , then : cat .boto3_credentials

Warning: each call of AllasStorage with .csc_creds.json OVERRIDES the .boto3_credentials (i.e. those are not usable anymore)


"""


# Initialize a storage without arguments (env vars ALLAS_PASSWORD and ALLAS_USERNAME must be defined)
# allas_storage = ldm.AllasStorage()
# Initiliaze a storage with AWS type credentials
# allas_storage = ldm.AllasStorage(".boto3_credentials")
# Initialize a storage with CSC credentials
allas_storage = ldm.AllasStorage(".csc_creds.json", timeout=100, retries=3)

# show allas projects associated with the csc account
print(allas_storage.projects_names)

# list buckets for a specific project from allas_storage
allas_storage.list_buckets("project_2012747")

# retrieve the S3 ressource corresponding to the project
s3_resource = allas_storage.get_s3_resource("project_2012747")

# # # getting bucket names
# # for bucket in s3_resource.buckets.all():
# #     print(bucket.name)

# # # uploading a file (key is the distant file)
# # s3_resource.Bucket("A7_LOBSTER").upload_file(Filename="local_file.txt", Key="distant_file.txt")

# Downloading a file
#s3_resource.Bucket("NewNasdaq").download_file(Filename="data/2025-10-09_snapshots_10.parquet", Key="LOBSTER/AAPL/2025-10-09_snapshots_10.parquet")

# # getting all existing items in the bucket
# bucket_objects = [obj.key for obj in s3_resource.Bucket("A7_LOBSTER").objects.all()]

# # use the allas_storage object to upload/download
# allas_storage.upload("kanniain", "A7_LOBSTER", "local_file.txt", "distant_file.txt")
# allas_storage.download("kanniain", "A7_LOBSTER", "local_file.txt", "distant_file.txt")


allas_storage.upload("project_2012747", "NewNasdaq", "data/features/features_TSLA_2025-12-17_messages_10_mmaps/image_array.uint8.mmap", "LOBSTER/TSLA/2025-12-17_order_images.uint8.mmap")

import os
import urllib.request
import hashlib
import tarfile

archive_name = "groove2groove-data-v1.0.0"
archive = archive_name + ".tar.gz"
download_url = "https://zenodo.org/record/3958000/files/" + archive + "?download=1"
checksum = "c407de7b3676267660c88dc6ee351c79"

if not os.path.exists(archive):
    urllib.request.urlretrieve(download_url, archive)
else:
    print(f"File {archive} already exists, skipping download")

with open(archive, "rb") as f:
    if hashlib.md5(f.read()).hexdigest() != checksum:
        raise ValueError(f"Checksum verification failed for {archive}")

for directory in ["train/fixed", "val/fixed", "test/fixed"]:
    if os.path.exists(directory):
        print(f"Removing {directory}")
        os.system("rm -rf " + directory)
    print(f"Extracting {directory}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=".", members=[m for m in tar.getmembers()
                                           if m.name.startswith(f"{archive_name}/midi/{directory}")])

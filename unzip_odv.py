import os
import zipfile
import glob

src_dir = "./odv"
dst_dir = "./odv/data"

for zip_path in glob.glob(os.path.join(src_dir, "*.zip")):
    folder_name = os.path.splitext(os.path.basename(zip_path))[0]
    extract_dir = os.path.join(dst_dir, folder_name)
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

print("Done.")

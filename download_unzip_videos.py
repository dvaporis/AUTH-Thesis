import re
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "owner/dataset-name"
DOWNLOAD_DIR = "./data"
MAX_WORKERS = 6

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

api = KaggleApi()
api.authenticate()

# Match files starting with 01 and ending with .mp4
pattern = re.compile(r"^01.*\.mp4$")

files = api.dataset_list_files(DATASET).files


def download_and_unzip(file_obj):
    name = file_obj.name

    if not pattern.match(name):
        return None

    try:
        print(f"⬇️ Downloading: {name}")
        
        # Download (Kaggle saves as .zip)
        api.dataset_download_file(
            DATASET,
            file_name=name,
            path=DOWNLOAD_DIR,
            force=True,
            quiet=True
        )

        zip_path = os.path.join(DOWNLOAD_DIR, f"{name}.zip")

        # Unzip
        if os.path.exists(zip_path):
            print(f"📦 Extracting: {name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_DIR)

            os.remove(zip_path)
            print(f"✅ Done: {name}")

        return name

    except Exception as e:
        print(f"❌ Failed: {name} | {e}")
        return None


# Parallel execution
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_and_unzip, f) for f in files]

    for future in as_completed(futures):
        result = future.result()
        if result:
            pass  # already logged

print("🎉 All matching files processed.")
import kagglehub
import os
import shutil

# 1. Download the dataset (this gets the local path where it's cached)
dataset_path = kagglehub.dataset_download("thbdh5765/audio-visual-database-of-emotional-speech-and-song")

# 2. Define where you want to move your specific files
target_dir = "./ravdess_videos_only"
os.makedirs(target_dir, exist_ok=True)

print(f"Scanning for files in: {dataset_path}")

# 3. Iterate through actor folders and find files starting with '01-'
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.startswith("01-") and file.endswith(".mp4"):
            source_file = os.path.join(root, file)
            # Move or copy the file to your target directory
            shutil.copy(source_file, os.path.join(target_dir, file))

print(f"Done! Your filtered videos are in: {target_dir}")
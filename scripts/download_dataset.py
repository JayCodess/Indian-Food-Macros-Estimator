import os
import kaggle
import zipfile

dataset_name = "iamsouravbanerjee/indian-food-images"
download_path = "data/"

print(f"Downloading dataset {dataset_name} to {download_path}...")
kaggle.api.authenticate()
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
print("Download and unzip complete!")

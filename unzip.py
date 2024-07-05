import zipfile
import os
import time
import sys

CSV_DIRECTORY = "/home/pouria/project/csv_files/currency_data.zip"

class UNZIP:
    def unzip_file(self, zip_path, extract_to='.'):
        if not os.path.exists(zip_path):
            # print(f"The file at {zip_path} does not exist.")
            return
        
        if not zipfile.is_zipfile(zip_path):
            # print(f"The file at {zip_path} is not a valid zip file.")
            return
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            # print(f"\rExtracted all contents of {zip_path} to {extract_to}", end="", flush=True)
        
        os.remove(zip_path)
        # print(f"\rDeleted the zip file: {zip_path}", end="", flush=True)
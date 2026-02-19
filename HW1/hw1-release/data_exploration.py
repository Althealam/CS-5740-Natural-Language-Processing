# Name(s):
# Netid(s):

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import zipfile
import os
import shutil
import csv

## ================ Helper functions for loading data ==========================

def unzip_file(zip_filepath, dest_path):
    """
    Returns boolean indication of whether the file was successfully unzipped.

    Input:
      zip_filepath: String, path to the zip file to be unzipped
      dest_path: String, path to the directory to unzip the file to
    Output:
      result: Boolean, True if file was successfully unzipped, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        return True
    except Exception as e:
        return False


def unzip_data(zipTarget, destPath):
    """
    Unzips a directory, and places the contents in the original zipped
    folder into a folder at destPath. Overwrites contents of destPath if it
    already exists.

    Input:
            None
    Output:
            None

    E.g. if zipTarget = "../dataset/student_dataset.zip" and destPath = "data"
          then the contents of the zip file will be unzipped into a directory
          called "data" in the cwd.
    """
    # First, remove the destPath directory if it exists
    if os.path.exists(destPath):
        shutil.rmtree(destPath)

    unzip_file(zipTarget, destPath)

    # Get the name of the subdirectory
    sub_dir_name = os.path.splitext(os.path.basename(zipTarget))[0]
    sub_dir_path = os.path.join(destPath, sub_dir_name)

    # Move all files from the subdirectory to the parent directory
    for filename in os.listdir(sub_dir_path):
        shutil.move(os.path.join(sub_dir_path, filename), destPath)

    # Remove the subdirectory
    os.rmdir(sub_dir_path)


def read_csv(filepath):
    """
    Reads an CSV file (premise, hypothesis, label) and returns a dictionary 
    where each key is the line number (starting from 0) and each value is a dictionary:
    {'premise': ..., 'hypothesis': ..., 'label': ...}
    """
    data = {'premise': [], 'hypothesis': [], 'label': []}
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'].strip() == '':
                continue  # skip empty rows
            data['premise'].append(row['premise'].strip())
            data['hypothesis'].append(row['hypothesis'].strip())
            data['label'].append(int(row['label'].strip()))
    
    return data
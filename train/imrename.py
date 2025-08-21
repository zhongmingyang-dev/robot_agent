import glob
import os
import sys


directory = "data"
counter = 651


if not os.path.exists(directory):
    print(f"Error: '{directory}' doesn't exist")
    sys.exit(0)

# 获取目录下所有文件（不包括子目录）
files = glob.glob(os.path.join(directory, '*'))
files = [f for f in files if os.path.isfile(f)]
files.sort()

for file_path in files:
    ext = os.path.splitext(file_path)[1]

    new_filename = f"{counter:012d}{ext}"
    new_path = os.path.join(directory, new_filename)

    try:
        os.rename(file_path, new_path)
        print(f"rename: {os.path.basename(file_path)} -> {new_filename}")
        counter += 1
    except Exception as e:
        print(f"Failed to rename {file_path}: {str(e)}")

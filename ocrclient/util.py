import os 
import shutil


def smart_make_dirs(dir_paths):
    if os.path.exists(dir_paths):
        shutil.rmtree(dir_paths)
    os.makedirs(dir_paths)
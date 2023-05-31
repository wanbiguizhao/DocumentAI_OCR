import os 
import shutil


def smart_make_dirs(dir_paths):
    if os.path.exists(dir_paths):
        shutil.rmtree(dir_paths)
    os.makedirs(dir_paths)
def is_contains_chinese(strs):
    for _char in strs:
        if not('\u4e00' <= _char <= '\u9fa5'):
            return False
    return True
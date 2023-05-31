import os
PROJECT_DIR= os.path.dirname(
        os.path.dirname(os.path.realpath( __file__))
    
)
APP_DIR= os.path.join(PROJECT_DIR,"ocrclient")
import json
JSON_DATA_PATH="tmp/0ocrdata/"
JSON_DATA_PATH="tmp/0ocrdata/data.json"
HAN_IMAGE_PATH="ocrclient/tmp/han_image_path.json"
CORUPS_PATH="ocrclient/tmp/sentences.json"
def dump_json_data(json_data,json_file_path):
    with open(json_file_path,"w") as ocr_data_file:
        json.dump(json_data,ocr_data_file,ensure_ascii=False,indent=2)
def load_json_data(json_file_path):
    with open(json_file_path,"r") as ocr_data_file:
        return json.load(ocr_data_file)

import shutil
basic_image_dir=os.path.join(PROJECT_DIR,"tmp/word2imgtop10_0530")
from util import smart_make_dirs
smart_make_dirs(basic_image_dir)
# 做一次汉字图片的归集word2map。
for han_name,path_list in   load_json_data(HAN_IMAGE_PATH).items():
    han_dir=os.path.join(basic_image_dir,f"@{han_name}")
    smart_make_dirs(han_dir)
    for one_path in  path_list:
        file_name=one_path.split("/")[-1]
        if os.path.exists(f"{han_dir}/file_name" ):
            continue
        shutil.copy(one_path,han_dir)

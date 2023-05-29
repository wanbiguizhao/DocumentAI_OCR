import os
PROJECT_DIR= os.path.dirname(
        os.path.dirname(os.path.realpath( __file__))
    
)
APP_DIR= os.path.join(PROJECT_DIR,"ocrclient")
import json
JSON_DATA_PATH="tmp/0ocrdata/"
JSON_DATA_PATH="tmp/0ocrdata/data.json"
HAN_IMAGE_PATH="ocrclient/tmp/han_image_path.json"
def dump_json_data(json_data,json_file_path):
    with open(json_file_path,"w") as ocr_data_file:
        json.dump(json_data,ocr_data_file,ensure_ascii=False,indent=2)
def load_json_data(json_file_path):
    with open(json_file_path,"r") as ocr_data_file:
        return json.load(ocr_data_file)

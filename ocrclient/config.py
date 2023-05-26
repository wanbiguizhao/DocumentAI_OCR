import json
JSON_DATA_PATH="tmp/0ocrdata/"
JSON_DATA_PATH="tmp/0ocrdata/data.json"
def dump_json_data(json_data,json_file_path):
    with open(json_file_path,"w") as ocr_data_file:
        json.dump(json_data,ocr_data_file,ensure_ascii=False,indent=2)
def load_json_data(json_file_path):
    with open(json_file_path,"r") as ocr_data_file:
        return json.load(ocr_data_file)
# 尝试使用easyocr识别图像。
import os
import easyocr 
from tqdm import tqdm
from glob import glob
import cv2 as cv
PROJECT_DIR= os.path.dirname(
        os.path.dirname(os.path.realpath( __file__))
    
)
import json



def run():
    reader=easyocr.Reader(["ch_tra"],gpu=False)
    BASE_IMAGE_DIR="tmp/project_ocrSentences"
    #DST_IMAGE_DIR="tmp/ocrSentences_resize"
    #width_list=[]
    for image_path in tqdm(
        sorted(glob(os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,"*","*.png"),recursive=True))[:100]
        ):
        result=reader.readtext(image_path)
        print(image_path,result)
def analysis_han_pipline():
    #统计一下ocr中有哪些汉字和频次。
    from collections import Counter
    han_counter=Counter()
    
    sentences_list=[]
    APP_DIR=os.path.join(PROJECT_DIR,"easyocr")
    for ocr_json_file_path in  tqdm(glob(f"{APP_DIR}/**/*/ocr.json",recursive=True)):
        with open(ocr_json_file_path,"r") as jf:
            json_data=json.load(jf)
            for area_ocr_data in json_data["ocr"]:
                text=area_ocr_data[1]
                sentences_list.append(text)
                han_counter.update(text)
    with open(f"{APP_DIR}/tmp/han_freq.json","w") as hf, open(f"{APP_DIR}/tmp/sentences.json","w") as sf:
        json.dump(sentences_list,sf,indent=2,ensure_ascii=False)
        json.dump(dict(han_counter), hf,indent=2,ensure_ascii=False)




if __name__=="__main__":
    analysis_han_pipline()
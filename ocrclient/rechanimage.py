# 使用各种各样的ocr识别图片
from collections import defaultdict
import os
import tempfile 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(os.path.realpath( __file__))
)
import cv2 as  cv
from tqdm import tqdm

import easyocr
import glob
from PIL import Image
import numpy as np
import shutil
import uuid
import json



easyocr_reader =easyocr.Reader(["ch_tra"],gpu=True) 
import datetime


def rec_han_easyocr(img_path):
    """
    当前一个字是一张图片，CTC识别单字准确率低
    img_path 图片的路径
    times 一个图片水平复制的次数
    """
    current_img=cv.imread(img_path)
    # hc,wc,_=current_img.shape
    # newImg=np.zeros((hc,wc*times,3),dtype=current_img.dtype)
    # for i in range(times):
    #     newImg[:,wc*i:wc*(i+1),]=current_img.copy()
    ocr_result=easyocr_reader.recognize(img_path,reformat=True)
    print(ocr_result)    
    return ocr_result




def get_all_han_image(base_image_dir,pattern):
    """
    找到所有的汉字的图片
    """
    Migrator().run()
    for glob_path in tqdm(sorted(glob.glob(f"{base_image_dir}/{pattern}",recursive=True))):
        current_img=cv.imread(glob_path)
        hc,wc,_=current_img.shape
        image_uuid=uuid.uuid3(uuid.NAMESPACE_DNS, glob_path).hex
        if len(HanImageInfo.find(HanImageInfo.uuid == image_uuid).all()):
            continue
        
        image_info=HanImageInfo(
            uuid=image_uuid,
            image_path=glob_path,
            bad=0,
            h=hc,
            w=wc
        )
        image_info.save()
import time
from redisdata import *

def use_easyocr():
    Migrator().run()
    while HanImageInfo.find(HanImageInfo.easyocr==0).count()>0:
        index=0
        for image_info in tqdm(HanImageInfo.find(HanImageInfo.easyocr==0).page()):
            index+=1
            easyocr=image_info.easyocr
            if not easyocr:
                image_path=image_info.image_path
                #easyocr_reader.readtext_batched()
                ocr_result=easyocr_reader.recognize(image_path,reformat=True)
                #ocr_result=easyocr_webservice(image_path)
                if not ocr_result:
                    continue
                ocr_result=easyocrResultData(
                    text= ocr_result[0][1],
                    score= ocr_result[0][2],
                    text_len=len(ocr_result[0][1]),
                    image_uuid=image_info.uuid,
                )
                image_info.easyocr=1
                image_info.easyocr_pk=ocr_result.pk
                ocr_result.save()
                image_info.save()
        #time.sleep(2)
        # image_data["easyocr"]={
        #         "text": ocr_result[0][1],
        #         "score": ocr_result[0][2],
        #         "len":len(ocr_result[0][1])
        # }
def use_easyocr_batch():
    import requests
    def easyocr_webservice(image_path_list):
        colab_url="http://ac92-34-90-68-225.ngrok-free.app/batch"
        image_list=[]
        for image_path in image_path_list: 
            image_list.append(("image",open(image_path, 'rb')))
        
        files = image_list
        model_results = requests.post(
                colab_url,files=files
            )
        if model_results.status_code==200:
            ocr_result=model_results.json()
            return ocr_result
        return []
    Migrator().run()
    count=HanImageInfo.find(HanImageInfo.easyocr==0).count()
    while count>0:
        image_info_list=HanImageInfo.find(HanImageInfo.easyocr==0).page()
        image_path_list=[image_info.image_path for image_info in image_info_list]
        ocr_result_list=easyocr_webservice(image_path_list)# 批量薅羊毛，解决网络延时问题。
        for index,ocr_web_result in enumerate(ocr_result_list):
            if  ocr_web_result:
                ocr_result=easyocrResultData(
                    text= ocr_web_result[0][1],
                    score= ocr_web_result[0][2],
                    text_len=len(ocr_web_result[0][1]),
                    image_uuid=image_info_list[index].uuid,
                )
                
                image_info_list[index].easyocr_pk=ocr_result.pk
                ocr_result.save()
                image_info_list[index].easyocr=1
                image_info_list[index].save()
            else:
                image_info_list[index].easyocr=-1
                image_info_list[index].save()
        count=HanImageInfo.find(HanImageInfo.easyocr==0).count()
        print(count)


def pipe_line01():
    # 找打某个图片
    #get_all_han_image(base_image_dir="easyocr/tmp/ocr_data",pattern="*/words/*.png")
    #get_all_han_image(base_image_dir="tmp/wordpics",pattern="/*/*.png")
    use_easyocr_batch()

def pipe_analy_easyocr():
    """
    分析ocr识别的数据，进行处理。
    """
    read_data_from_redis()
    pass 

if __name__=="__main__":
    pipe_line01()
    pipe_analy_easyocr()
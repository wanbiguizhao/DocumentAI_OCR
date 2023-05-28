from glob import glob
import os
#os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
from redisdata import Migrator,HanImageInfo,paddleocrResultData
from tqdm import tqdm
import json
from paddleocr import PaddleOCR
from config import load_json_data,dump_json_data
paddleocr = PaddleOCR(
        det=False,
        rec=True,
        lang="chinese_cht",
        #lang="ch",
        cls=False,
        use_gpu=True,
        show_log=False
        
    )
#os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
from typing import Optional
from pydantic import EmailStr

from redis_om import (
    Field,
    HashModel,
    Migrator
)
class HanImageInfo(HashModel):
    uuid: str  = Field(index=True)
    image_path: str
    h: int
    w: int
    bad: Optional[int]
    easyocr:int= Field(index=True,default=0)
    easyocr_pk: Optional[str]
    paddleocr:int= Field(index=True,default=0)
    paddleocr_pk: Optional[str]
class easyocrResultData(HashModel):
    image_uuid:str= Field(index=True)
    score: float
    text: str
    text_len: int
class paddleocrResultData(HashModel):
    image_uuid:str= Field(index=True)
    score: float
    text: str
    text_len: int
def use_paddleocr():

    Migrator().run()
    count=HanImageInfo.find(HanImageInfo.paddleocr==0).count()
    while count>0:
        index=0
        for image_info in tqdm(HanImageInfo.find(HanImageInfo.paddleocr==0).all()):
            image_path=image_info.image_path
            #"tmp/ocrSentences/955-15_101/955-15_101_00.png"
            ocr_result=paddleocr.ocr(image_path,det=False,cls=False)
            
            if len(ocr_result[0])>0 and len(ocr_result[0][0])>0:
                #ocr_result=ocr_result[0]
                index+=1

                ocr_result=paddleocrResultData(
                    text= ocr_result[0][0][0],
                    score= ocr_result[0][0][1],
                    text_len=len(ocr_result[0][0][0]),
                    image_uuid=image_info.uuid,
                )
                image_info.paddleocr=1
                image_info.paddleocr_pk=ocr_result.pk
                ocr_result.save()
                image_info.save()
            else:
                image_info.paddleocr=-1
                image_info.save()
        count=HanImageInfo.find(HanImageInfo.paddleocr==0).count()
        print(count,index/100.0)
        import time
        time.sleep(10)

def test_cg_infer_image():
    image_list=sorted(glob("/home/liukun/gan/cg-gan-custom/tmp/infer_images/*.png"))
    index=0
    count=0
    same_count=0
    different_count=0
    diffu_cuont=0
    while index+1<len(image_list):
        pre_data=paddleocr.ocr(image_list[index],det=False,cls=False)
        ocr_result=paddleocr.ocr(image_list[index+1],det=False,cls=False)
        image_path=image_list[index+1]
        if ocr_result[0][0][0]==pre_data[0][0][0]:
            if ocr_result[0][0][1]> pre_data[0][0][1]:
                print("==="*5)
                print(image_list[index+1],ocr_result[0][0] )
                print(image_list[index],pre_data[0][0])
                count+=1
            same_count+=1
        else:
            if ocr_result[0][0][1]>0.3:
                print("==="*5)
                print(image_list[index+1],ocr_result[0][0] )
                print(image_list[index],pre_data[0][0])
                diffu_cuont+=1
                #count+=1

            different_count+=1



        index+=2
    print(count*2.0/len(image_list),same_count*2.0/len(image_list),different_count/len(image_list),diffu_cuont/len(image_list))
            

def ocr_infer_image(base_dir):
    os.listdir(base_dir)


def pipe_line01():
    use_paddleocr()
def ocr_infer_images():
    basic_infer_dir="tmp/infer_images"
    for uuid_image_dir in tqdm(os.listdir(basic_infer_dir)):
        file_list= os.listdir(f"{basic_infer_dir}/{uuid_image_dir}")

        file_list=[f for f in file_list if "png" in f or "pocr" in f]
        if len(file_list)!=2:
            print(uuid_image_dir)
            continue
            
        ocr_data={
            "paddleocr":{
                file_list[0]:paddleocr.ocr(f"{basic_infer_dir}/{uuid_image_dir}/{file_list[0]}",det=False,cls=False),
                file_list[1]:paddleocr.ocr(f"{basic_infer_dir}/{uuid_image_dir}/{file_list[1]}",det=False,cls=False)
            }
        }
        with open(f"{basic_infer_dir}/{uuid_image_dir}/pocr.json","w") as jsonf:
            json.dump(ocr_data,jsonf,ensure_ascii=False,indent=2)
if __name__=="__main__":
    os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
    #pipe_line01()
    ocr_infer_images()
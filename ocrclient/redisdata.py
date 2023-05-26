from collections import defaultdict

import os

from tqdm import tqdm
os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
from config import load_json_data,JSON_DATA_PATH
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


print(HanImageInfo.find(HanImageInfo.paddleocr==0).count())

def read_data_from_redis():
    # 从redis中拿到识别的数据
    # 找到字符长度为1，并且正确率达到80以上的。
    han_image_ocr_info=defaultdict(list)
    image_dict_info=load_json_data(JSON_DATA_PATH)
    ocr_count=0
    usefull_ocr_count=0
    for image_uuid in tqdm(image_dict_info.keys()):
        ocr_data=easyocrResultData.find(easyocrResultData.image_uuid==image_uuid).all()
        if not len(ocr_data)==0: 
            ocr_count+=1
            ocr_data=ocr_data[0]
            if ocr_data.score>0.65 and ocr_data.text_len==1 :
                usefull_ocr_count+=1

                han_image_ocr_info[ocr_data.text].append(
                    [
                        ocr_data.score,image_uuid
                    ]
                )
        ocr_data=paddleocrResultData.find(paddleocrResultData.image_uuid==image_uuid).all()
        if not len(ocr_data)==0:
            continue
        else:
            ocr_data=ocr_data[0]
            if ocr_data.score>0.65 and ocr_data.text_len==1 :
                han_image_ocr_info[ocr_data.text].append(
                    [
                        ocr_data.score,image_uuid
                    ]
                )

    print(ocr_count,
        len(han_image_ocr_info.keys()),usefull_ocr_count
    )
    print( sorted([ (len(v),k) for  k,v in han_image_ocr_info.items()]) )
def pipe_read_data():
    
    read_data_from_redis()

if __name__=="__main__":
    os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
    pipe_read_data()
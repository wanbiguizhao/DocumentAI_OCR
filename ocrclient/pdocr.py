import os
os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
from redisdata import Migrator,HanImageInfo,paddleocrResultData
from tqdm import tqdm
from paddleocr import PaddleOCR
paddleocr = PaddleOCR(
        det=False,
        rec=True,
        lang="chinese_cht",
        cls=False,
        use_gpu=True,
        show_log=False
        
    )
os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
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


def pipe_line01():
    use_paddleocr()

if __name__=="__main__":
    os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
    pipe_line01()
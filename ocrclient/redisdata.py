from collections import defaultdict

import os

from tqdm import tqdm
os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
from config import load_json_data,JSON_DATA_PATH
from typing import Optional
from pydantic import EmailStr
from redis_om import get_redis_connection

redis_conn = get_redis_connection()
def is_contains_chinese(strs):
    for _char in strs:
        if not('\u4e00' <= _char <= '\u9fa5'):
            return False
    return True
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
class hanResultData(HashModel):
    # ocr 识别的汉字个数是1的数据,放到数据库中进行分析
    image_uuid:str= Field(index=True)
    score: float= Field(index=True,sortable=True)
    from_ocr:str=Field(index=True)
    text: str=Field(index=True)# 汉字要是一个唯一的索引。
    cg_gan_text:str=Field(index=True,default="") # cg gan 识别的文字
    cg_gan_score:str=Field(index=True,default=-1.0,sortable=True)# cg gan 识别的分数。

    def info(self):
        return f"{self.text}\t{self.score}\t{self.from_ocr}\t{self.get_image_path()}"
    def get_image_path(self):
        return HanImageInfo.find(HanImageInfo.uuid==self.image_uuid).first().image_path
class hanData(HashModel):
    # 汉字的一般数据
    han:str= Field(index=True)# 汉字
    count: Optional[int]= Field(index=True,sortable=True,default=-1)# 识别出来的汉字

    def info(self):
        return f"{self.han}\t{self.count}"



print(HanImageInfo.find(HanImageInfo.paddleocr==0).count())

def move_usefull_ocr_data():
    # 对于识别后的ocr数据存储到hanResultData
    # 存储条件text的长度是1
    Migrator().run()
    def add_han_result_data(query_uuid,queryclass,queryclass_name):
        # 去hanResultData中查询是否有对应图片干净的ocr数据，如果有了返回，没有就更新。
        query_count=hanResultData.find(hanResultData.image_uuid==query_uuid , hanResultData.from_ocr==queryclass_name).count()
        if query_count>0:
            return
        ocr_data_all=queryclass.find(queryclass.image_uuid==query_uuid).all()
        if len(ocr_data_all)<1:
            return
        ocr_data=ocr_data_all[0]
        if ocr_data.text_len!=1:# 必须是一个汉字
            return
        hanRD=hanResultData(
            image_uuid=query_uuid,
            score=ocr_data.score,
            text=ocr_data.text,
            from_ocr=queryclass_name
        )
        hanRD.save()
        text=ocr_data.text
        if not is_contains_chinese(text):
                text="##"+text
        if hanData.find(hanData.han==text).count()==0:
            #print(ocr_data.text)
            #text=ocr_data.text
            hanData(
                han=text
            ).save()


    image_dict_info=load_json_data(JSON_DATA_PATH)# 加载之前所有的图片信息。
    for query_uuid in tqdm(image_dict_info.keys()):
        add_han_result_data(query_uuid,easyocrResultData,"easyocr")
        add_han_result_data(query_uuid,paddleocrResultData,"paddleocr")
def get_all_han():
    # 获得所有的汉字
    print(hanData.find().count())
    all_han_data=hanData.find().all()
    for han in all_han_data:
        if not is_contains_chinese(han.han) and "##" not in han.han:
            hanData.delete(han.pk)
    print(len([han.han for han in all_han_data if "##" not in han.han]))

def get_han_from_dir():
    # 解析文件路径，找到汉字
    #word2imgtop10
    #|-@
    #|-@週
    han_set=set()
    for dirname in os.listdir("tmp/word2imgtop10"):
        han=dirname.split("@")[-1]
        if is_contains_chinese(han):
            han_set.update(han)
    return han_set


def getallHan(n=10):
    # 根据汉字的分数拿到从高到底的汉字对应图片
    print("han count:",hanData.find().count()) 
    all_han_data=hanData.find().all()
    dir_han_set=get_han_from_dir()
    record_file=open("image_low_score_info.txt","w")
    for han in all_han_data:
        if "##" not in han.han and han.han not in dir_han_set:
            #han.count=hanResultData.find(hanResultData.text==han.han).count()
            # han.save()
            han_data_list=hanResultData.find(hanResultData.text==han.han).sort_by("-score").page(0,1)
            for han_data in han_data_list:
                record_file.write(han_data.info()+"\n")
        # else:
        #     #han.count=-10
        #     han.save()

    record_file.close()
def getTopNHan(hantext,n=10):
    # 根据汉字的分数拿到从高到底的汉字对应图片
    print("han count:",hanData.find().count()) 
    #all_han_data=hanData.find().all()
    record_file=open("image_low_score_info.txt","w")

    han_data_list=hanResultData.find(hanResultData.text==hantext).sort_by("-score").page(0,limit=10)
    for han_data in han_data_list:
        record_file.write(han_data.info()+"\n")
        print(han_data.info())
    record_file.close()

def getHanByCount(count=10):
    han_data_list=hanData.find(hanData.count<count,hanData.count>0).sort_by("count").all()
    for hd in han_data_list:
        print(hd.info())
    print(len(han_data_list))
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


def pipline_template_topn():
    """
    把汉字中，展示最好的10个图片，
    """

    pass 
if __name__=="__main__":
    os.environ.setdefault("REDIS_OM_URL","redis://:wwheqq@0.0.0.0:6379")
    #move_usefull_ocr_data()
    #getTopNHan("群")
    #getHanByCount()
    getallHan()
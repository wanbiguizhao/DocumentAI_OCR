from collections import defaultdict

import os

from tqdm import tqdm
from config import load_json_data,JSON_DATA_PATH
from typing import Any, Optional
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
    Migrator,
    JsonModel
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
    cg_gan_text:str=Field(index=True,default="") # cg gan 识别的文字,这里先直接使用print2write的数据。
    cg_gan_score:float=Field(index=True,default=-1.0,sortable=True)# cg gan 识别的分数。
    cg_gan_origin_text:str=Field(index=True,default="") # cg gan 识别的文字,这里先直接使用print2write的数据。这里使用的是
    cg_gan_origin_score:float=Field(index=True,default=-1.0,sortable=True)# cg gan 识别的分数。
    def info(self):
        return f"{self.text}\t{self.score}\t{self.from_ocr}\t{self.get_image_path()}\t{self.image_uuid}"
    def get_image_path(self):
        return HanImageInfo.find(HanImageInfo.uuid==self.image_uuid).first().image_path
    def get_text_score_dict(self):
        text_dict=defaultdict(float)
        text_dict[self.text]+=self.score
        text_dict[self.cg_gan_origin_text]=text_dict[self.cg_gan_origin_text]+self.cg_gan_origin_score*0.5
        text_dict[self.cg_gan_text]+=self.cg_gan_score
        return text_dict
        
class tempHanImage(HashModel):
    # ocr 识别的汉字个数是1的数据,放到数据库中进行分析
    image_uuid:str= Field(index=True)
    score: float= Field(index=True,sortable=True)
    text:str=Field(index=True)
    def info(self):
        return f"{self.text}\t{self.score}\t{self.image_uuid}\t{self.get_image_path()}"
    def get_image_path(self):
        return HanImageInfo.find(HanImageInfo.uuid==self.image_uuid).first().image_path
class hanData(HashModel):
    # 汉字的一般数据
    han:str= Field(index=True)# 汉字
    count: Optional[int]= Field(index=True,sortable=True,default=-1)# 识别出来的汉字
    def info(self):
        return f"{self.han}\t{self.count}"
    @classmethod
    def make_key(cls, part: str):
        global_prefix = getattr(cls._meta, "global_key_prefix", "").strip(":")
        model_prefix = "__main__.hanData"
        return f"{global_prefix}:{model_prefix}:{part}"



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

def get_han_from_dir(min_num=10):
    # 解析文件路径，找到汉字，min_num要求汉字里面最少包含min_num个图片，才算有效汉字
    #word2imgtop10
    #|-@
    #|-@週
    han_set=set()
    for dirname in os.listdir("tmp/word2imgtop10"):
        han=dirname.split("@")[-1]
        if is_contains_chinese(han) and len(os.listdir(f"tmp/word2imgtop10/{dirname}"))>=min_num:# 超过10个图片
        
            han_set.update(han)
    return han_set


def getallHan(n=5):
    # 根据汉字的分数拿到从高到底的汉字对应图片
    print("han count:",hanData.find().count()) 
    all_han_data=hanData.find().all()
    dir_han_set=get_han_from_dir()
    record_file=open("image_top_score_info.txt","w")
    for han in all_han_data:
        if "##" not in han.han and han.han not in dir_han_set:
            han_data_list=tempHanImage.find(tempHanImage.text==han.han).sort_by("-score").page(0,n)
            for han_data in han_data_list:
                record_file.write(han_data.info()+"\n")
    record_file.close()
def getNtempHanImage(page_size=10,han_num=30):
    Migrator().run()
    all_han_data=hanData.find().all()
    print(len(all_han_data))
    dir_han_set=get_han_from_dir()
    result_dict=defaultdict(list)
    for han in all_han_data:
        if "##" not in han.han and han.han not in dir_han_set:
            han_data_list=tempHanImage.find(tempHanImage.text==han.han).sort_by("-score").page(0,page_size)
            for han_data in han_data_list:
                result_dict[han_data.text].append(
                {
                    "score":han_data.score,
                    "image_uuid":han_data.image_uuid,
                    "image_path":han_data.get_image_path(),
                    "han":han_data.text
                })
        if len(result_dict)>=han_num:
            break
    return result_dict
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

def dump_han_ocr_data():
    #从hanResultData检索图片对应的最有可能的汉字，然后打印出来信息。
    image_dict_info=load_json_data(JSON_DATA_PATH)
    record_file=open("image_ocr_info.txt","w")
    for image_uuid in tqdm(image_dict_info.keys()):
        ocr_data=hanResultData.find(hanResultData.image_uuid==image_uuid).sort_by("-score").all()
        if len(ocr_data)>0:
            ocr_data=ocr_data[0]
            record_file.write(ocr_data.info()+"\n")
    record_file.close()
def use_cg_gan_ocr_data():
    # 使用cggan生成的数据进行数据分析。
    Migrator().run()
    basic_infer_dir="tmp/infer_images"
    for uuid_image_dir in tqdm(os.listdir(basic_infer_dir)):
        if os.path.exists(f"{basic_infer_dir}/{uuid_image_dir}/ocr.json"):
            easy_ocr=load_json_data(f"{basic_infer_dir}/{uuid_image_dir}/ocr.json")
            ocr_data=hanResultData.find(hanResultData.image_uuid==uuid_image_dir,hanResultData.from_ocr=="easyocr").all()
            if len(ocr_data)==1:
                ocr_data=ocr_data[0]
                for k,v in easy_ocr["easyocr"].items():
                    if "_img_print.png" in k:
                        #print(easy_ocr["easyocr"][k][0][1],easy_ocr["easyocr"][k][0][2],k,"origin_text")
                        ocr_data.cg_gan_origin_text=easy_ocr["easyocr"][k][0][1]
                        ocr_data.cg_gan_origin_score=easy_ocr["easyocr"][k][0][2]
                    else:
                        #print(easy_ocr["easyocr"][k][0][1],easy_ocr["easyocr"][k][0][2],k,"text")
                        ocr_data.cg_gan_text=easy_ocr["easyocr"][k][0][1]
                        ocr_data.cg_gan_score=easy_ocr["easyocr"][k][0][2]
                ocr_data.save()
            # key1=easy_ocr["easyocr"].keys()[0],key2=easy_ocr["easyocr"].keys()[1]
            # if "_img_print.png" in key1:
            #     print(easy_ocr["easyocr"][key1][0][1],key1,"origin_text")
            #     print(easy_ocr["easyocr"][key2][0][1],key2,"text")
        if os.path.exists(f"{basic_infer_dir}/{uuid_image_dir}/pocr.json"):
            paddle_ocr=load_json_data(f"{basic_infer_dir}/{uuid_image_dir}/pocr.json")
            ocr_data=hanResultData.find(hanResultData.image_uuid==uuid_image_dir,hanResultData.from_ocr=="paddleocr").all()
            if len(ocr_data)==1:
                ocr_data=ocr_data[0]
                for k,v in paddle_ocr["paddleocr"].items():
                    if "_img_print.png" in k:
                        #print(paddle_ocr["paddleocr"][k][0][0][0],paddle_ocr["paddleocr"][k][0][0][1],k,"origin_text")
                        ocr_data.cg_gan_origin_text=paddle_ocr["paddleocr"][k][0][0][0]
                        ocr_data.cg_gan_origin_score=paddle_ocr["paddleocr"][k][0][0][1]
                    else:
                        #print(paddle_ocr["paddleocr"][k][0][0][0],paddle_ocr["paddleocr"][k][0][0][1],k,"text")
                        ocr_data.cg_gan_text=paddle_ocr["paddleocr"][k][0][0][0]
                        ocr_data.cg_gan_score=paddle_ocr["paddleocr"][k][0][0][1]
                ocr_data.save()



def analysis_hanResultData():
    # 分析数据hanResultData 找到所有的汉字
    """
    把汉字中，展示最好的10个图片，
    """
    Migrator().run()
    han_set=set()
    image_dict_info=load_json_data(JSON_DATA_PATH)
    for image_uuid in tqdm(image_dict_info.keys()):
        ocr_data_list=hanResultData.find(hanResultData.image_uuid==image_uuid).sort_by("-score").all()
        han_score_dict=defaultdict(float)# 拿到一张图片对应的所有的结果
        if len(ocr_data_list)>0:
            for ocr_data in ocr_data_list:
                for k,v in ocr_data.get_text_score_dict().items():
                    han_score_dict[k]+=v 
        #score_han_list=ocr_data.get_top_text()

        for han,score in han_score_dict.items():
            if score<0.5 or not is_contains_chinese(han):
                continue
            han_set.update(han)
            if tempHanImage.find(tempHanImage.image_uuid==image_uuid,tempHanImage.text==han,tempHanImage.score==score).count()==0:
                tempHanImage(
                    image_uuid=image_uuid,
                    text=han,
                    score=score
                ).save()    
    print(len(han_set))

def dump_han_top10score_tempHanImage():
    pass 



def pipline_template_topn():
    pass 
if __name__=="__main__":
    #use_cg_gan_ocr_data()
    #move_usefull_ocr_data()
    #getTopNHan("群")
    #getallHan()
    #analysis_hanResultData()
    #getallHan()
    getNtempHanImage(page_size=10,han_num=30)
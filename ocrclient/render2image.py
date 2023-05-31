
import random

from tqdm import tqdm
from config import load_json_data,dump_json_data,HAN_IMAGE_PATH,CORUPS_PATH,APP_DIR
import cv2 as  cv
import numpy as np
from util import smart_make_dirs


class WordImgSet:
    """
    汉字对应的图片
    """
    def __init__(self,han_image_path_data=load_json_data(HAN_IMAGE_PATH),cropimage=True) -> None:
        self.han_info={}

        for han,path_list in tqdm(han_image_path_data.items(),desc="加载图片数据中"):
            images=[]
            for image_path in path_list:
                img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)# 灰度图
                h,w=img.shape
                if h>75:
                    continue 
                if cropimage==True:# 有的图片本身就已经crop过了。
                    images.append( self.cropImage(img) )
                else:
                    images.append( img)
            if len(images)>0:
                self.han_info[han]={
                    "images":images,
                    "index":0
                }
        self.han_list=list(self.han_info.keys())
    def cropImage(self,image):
        # 去掉白边
        h,w=image.shape
        blur = cv.GaussianBlur(image,(5,5),0)
        ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        l,r,u,d=0,w-1,h-1,0
        while  l<r and sum(th_image[:,l])==255*h:
            l+=1
        while l<r and sum(th_image[:,r])==255*h:
            r-=1
        while d<u and sum(th_image[d,:])==255*w:
                d+=1
        while d<u and sum(th_image[u,:])>=255*w*0.98:
            u-=1
        return image[ d : u+1, max(l-1,0):min( w,r+1+1)] # 上下左右各留了2像素
    
    def text2image(self,text,random_mis_char=False):
        # 对于text中不存在的字符，是否随机替换一个字符
        ret_text=[]
        guess_hd,guess_hu=75,0
        max_width=96*len(text)
        sentence_img=np.zeros((75,max_width),dtype = np.uint8)
        guess_max_h=0
        sentence_img[:,:]=255
        beg_w=0
        for ch in text:
            if ch not in self.han_info:
                if not random_mis_char:
                    continue
                ch=random.choice(self.han_list)# 随机从库中拿一个字符
            ret_text.append(ch)
            wordimg=self.get_han_image(ch)
            h,w=wordimg.shape
            
            h_mid=75//2
            whd,whu= (h_mid-h//2),(h_mid+h-h//2)#计算汉字的上下界线
            guess_hu=max(guess_hu,whu)
            guess_hd=min(guess_hd,whd)
            sentence_img[whd:whu,beg_w:beg_w+w]=wordimg.copy()# 这一步非常重要，决定了图片的汉字是以下对齐
            beg_w+=w 
        if len(ret_text)==0:
            return None,[]
        return cv.cvtColor(sentence_img[guess_hd:guess_hu+1,:beg_w], cv.COLOR_GRAY2BGR),ret_text

    def get_han_image(self,ch):
        index=self.han_info[ch]["index"]
        wordimg=self.han_info[ch]["images"][index]
        self.han_info[ch]["index"]=(index+1)%len(self.han_info[ch]["images"])# 指针指向下一个图片
        return wordimg
    def get_random_image(self):
        # 随机返回一个字符
        ch=random.choice(self.han_info.keys())
        index=self.han_info[ch]["index"]
        wordimg=self.han_info[ch]["images"][index]
        self.han_info[ch]["index"]=(index+1)%len(self.han_info[ch]["images"])# 指针指向下一个图片
        return wordimg,ch
def save_image(image_name,gray_image):
    cv.imwrite(image_name,gray_image)


def load_corups():
    # 加载训练语料数据。
    rmrb_corups=[]
    with open("tmp/all-1946-1956-rmrb.txt") as rmrb_corups_file:
        line_content=rmrb_corups_file.readline()
        while line_content:
            line_content=line_content.replace("\n","")
            if len(line_content)>0:
                rmrb_corups.append(line_content)
            line_content=rmrb_corups_file.readline()
    return rmrb_corups


def pipeline_train_val_data():
    # 生成训练数据集和验证数据集
    han_image_path_data=load_json_data(HAN_IMAGE_PATH)
    wis=WordImgSet(han_image_path_data=han_image_path_data,cropimage=False)
    corups=load_corups()
    pipelinetrain(wis,corups)
    pipelineval(wis,corups)

def pipelinetrain(wis:WordImgSet,corups):
    #批量生成用于训练的语料。
    han_image_path_data=load_json_data(HAN_IMAGE_PATH)
    multiple_num=2
    
    images_save_dir=f"{APP_DIR}/tmp/traindata/images"
    rec_gt_train_path=f"{APP_DIR}/tmp/traindata/rec_gt_train.txt"
    smart_make_dirs(images_save_dir)
    
    #corups=load_json_data(CORUPS_PATH)
    random.shuffle(corups)
    rec_gt_train_file=open(rec_gt_train_path,"w")
    for index,corup in tqdm(enumerate(corups),desc="生成训练数据集"):
        png_image_list=[]
        for small_index in range(multiple_num):
            grayimage,corup_text=wis.text2image(corup,random_mis_char=False)
            if grayimage is not None:
                png_name=f"{index}-{small_index}.png"
                save_image(f"{images_save_dir}/{png_name}",grayimage)
                png_image_list.append(f'"{png_name}"')
        if len(png_image_list)>0:
            rec_gt_train_file.write(
                "["+",".join(png_image_list)+"]"+"\t"+"".join(corup_text)+"\n"
            )
    rec_gt_train_file.close()
def pipelineval(wis:WordImgSet,corups):
    #批量生成用于验证数据。
    #han_image_path_data=load_json_data(HAN_IMAGE_PATH)
    multiple_num=1
    
    images_save_dir=f"{APP_DIR}/tmp/valdata/images"
    rec_gt_train_path=f"{APP_DIR}/tmp/valdata/rec_gt_val.txt"
    smart_make_dirs(images_save_dir)
    #wis=WordImgSet(han_image_path_data=han_image_path_data)
    #corups=load_json_data(CORUPS_PATH)
    random.shuffle(corups)
    rec_gt_train_file=open(rec_gt_train_path,"w")
    for index,corup in tqdm(enumerate(corups),desc="生成验证数据集")[:1500]:
        #png_image_list=[]
        for small_index in range(multiple_num):
            grayimage,corup_text=wis.text2image(corup,random_mis_char=True)
            if grayimage is not None:
                png_name=f"{index}-{small_index}.png"
                save_image(f"{images_save_dir}/{png_name}",grayimage)
                rec_gt_train_file.write(
                png_name+"\t"+"".join(corup_text)+"\n"
            )
        #         png_image_list.append(f'"{png_name}"')
        # if len(png_image_list)>0:
            # rec_gt_train_file.write(
            #     "["+",".join(png_image_list)+"]"+"\t"+"".join(corup_text)+"\n"
            # )
    rec_gt_train_file.close()
if __name__=="__main__":
    #load_corups()
    pipeline_train_val_data()
    # han_image_path_data=load_json_data(HAN_IMAGE_PATH)# 存放图片，汉字对应的图片
    # wis=WordImgSet(han_image_path_data=han_image_path_data)
    # wis.text2image("加强互助合作生產及代耕工作中取得的經驗和存在的問題")
    # print("asdf")

    




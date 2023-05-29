

from random import random

from tqdm import tqdm
from config import load_json_data,dump_json_data,HAN_IMAGE_PATH
import cv2 as  cv
import numpy as np
han_image_path_data=load_json_data(HAN_IMAGE_PATH)# 存放图片，汉字对应的图片


class WordImgSet:
    """
    汉字对应的图片
    """
    def __init__(self,han_image_path_data=han_image_path_data) -> None:
        self.han_info={}
        for han,path_list in tqdm(han_image_path_data.items(),desc="加载图片数据中"):
            images=[]
            for image_path in path_list:
                img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)# 灰度图
                h,w=img.shape
                if h>75:
                    continue 

                images.append( self.cropImage(img) )
            if len(images)>0:
                self.han_info[han]={
                    "images":images,
                    "index":0
                }
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

    def text2image(self,text):
        guess_max_h=0
        max_width=96*len(text)
        sentence_img=np.zeros((75,max_width),dtype = np.uint8)
        sentence_img[:,:]=255
        beg_w=0
        for ch in text:
            if ch not in self.han_info:
                continue
            wordimg=self.get_han_image(ch)
            h,w=wordimg.shape
            guess_max_h=max(guess_max_h,h)
            sentence_img[75-h:75,beg_w:beg_w+w]=wordimg.copy()# 这一步非常重要，决定了图片的汉字是以下对齐
            beg_w+=w 
        grayImage = cv.cvtColor(sentence_img[75-guess_max_h:,:beg_w], cv.COLOR_GRAY2BGR)
        cv.imwrite("text.png",grayImage)
    def get_han_image(self,ch):
        index=self.han_info[ch]["index"]
        wordimg=self.han_info[ch]["images"][index]
        self.han_info[ch]["index"]=(index+1)%len(self.han_info[ch]["images"])# 指针指向下一个图片
        return wordimg
if __name__=="__main__":
    wis=WordImgSet(han_image_path_data=han_image_path_data)
    wis.text2image("加互助合作生產及代耕工作中取得的經驗和存在的問題")
    print("asdf")

    




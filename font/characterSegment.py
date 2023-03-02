
import os 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(__file__)
)
import sys 
sys.path.append(PROJECT_DIR)

import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np 
import glob
from tqdm import tqdm


# 可视化原图



def displayCharacter(img_path):
    print(img_path)
    plt.figure()
    # timg=cv.imread('tmp/ocrSentences/1955-03_03/1955-03_03_00.png')
    timg=cv.imread(img_path)
    print(timg.shape)
    h,w,_=timg.shape
    word_flag=False
    i=1
    blank_size=0
    for wi in range(w):
        meanv=np.mean(timg[:,wi,:],axis=(0,1))
        #print(wi,meanv)
        if word_flag==False :
            if  np.mean(timg[:,wi,:],axis=(0,1))<254:
                word_flag=True
                bw=wi 
        else:
            #print(bw,wi)
            
            if  meanv >250:
                if blank_size >1:
                    
                    word_flag=False
                    seg_img=timg[:,max(0,bw-2):min(wi+2,w),:]
                    #print(seg_img)
                    plt.subplot(6,6,i)
                    plt.imshow(seg_img)
                    i+=1
                    blank_size=0
                else:
                    blank_size+=1
    plt.show(block=True)

def segAndSaveWordPics(baseDir="tmp/wordpics"):
    # 把所有的图片切割成单独的汉字
    # 每切割2000个字，就单独的组建一个文件夹00001
    pic_count=1
    COUNT_MAX_PIC=2000
    for pngf in tqdm(
        glob.glob(os.path.join(PROJECT_DIR,'tmp','*ocr*','**','*.png'),recursive=True )
                        ):
        timg=cv.imread(pngf)
        h,w,_=timg.shape
        word_flag=False
        i=1
        blank_size=0
        for wi in range(w):
            meanv=np.mean(timg[:,wi,:],axis=(0,1))
            #print(wi,meanv)
            if word_flag==False :
                if  np.mean(timg[:,wi,:],axis=(0,1))<254:
                    word_flag=True
                    bw=wi 
            else:
                #print(bw,wi)
                if  meanv >250:
                    if blank_size >1:
                        
                        word_flag=False
                        seg_img=timg[:,max(0,bw-2):min(wi+2,w),:]
                        pic_name=str(pic_count).rjust(6,'0')+".png"
                        
                        pic_dir=str((pic_count//COUNT_MAX_PIC)*COUNT_MAX_PIC+1).rjust(6,'0')
                        abs_pic_dir=os.path.join(baseDir,pic_dir)
                        if not os.path.exists(abs_pic_dir ):
                            os.makedirs(abs_pic_dir)

                        cv.imwrite( os.path.join(abs_pic_dir,pic_name)  ,seg_img)
                        pic_count+=1
                        #print(seg_img)
                        #plt.subplot(6,6,i)
                        #plt.imshow(seg_img)
                        i+=1
                        blank_size=0
                    else:
                        blank_size+=1
        



def show_all_png():
    #print(PROJECT_DIR)
    for pngf in tqdm(
        glob.glob(os.path.join(PROJECT_DIR,'tmp','*ocr*','**','*02*.png'),recursive=True )
                        ):
        displayCharacter(pngf)

if __name__=="__main__":
    #merge_labels_info(DATA_DIR,save_dir=os.path.join(PROJECT_DIR,'tmp'))
    #in_f=os.path.join(PROJECT_DIR,'tmp','1954-01.pdf')
    #show_all_png()
    segAndSaveWordPics()
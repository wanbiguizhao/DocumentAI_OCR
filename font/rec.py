"""
调用paddle，为每个字找打对应的图片
"""
import os
import tempfile 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(__file__)
)
import cv2 as  cv
from tqdm import tqdm
from paddleocr import PaddleOCR
import glob
from PIL import Image
import numpy as np
ocr = PaddleOCR(
    det=False,
    rec=True,
    lang="chinese_cht",
    cls=False
)  
# DEBUG=True
# image_path="/home/liukun/ocr/DocumentAI_OCR/tmp/wordpics/000001/000002.png"
# pre_image_path=""
# for pngf in tqdm(glob.glob(os.path.join(PROJECT_DIR,'tmp','*word*','**','*.png'),recursive=True )):
#     if pre_image_path:
#         pre_img=cv.imread(pre_image_path)
#         hp,wp,_=pre_img.shape
#         current_img=cv.imread(pngf)
#         hc,wc,_=current_img.shape
#         merge_image=np.zeros((max(hp,hc),wp+wc,3),dtype=int)
#         merge_image[:,:,:]=255
#         merge_image[0:hp,0:wp,:]=pre_img
#         merge_image[0:hc,wp:,:]=current_img
#         #print(pre_img.shape,current_img.shape)
#         ocr_result=ocr.ocr(merge_image)
#         if len(ocr_result[0])>0:
#             print(pngf,ocr_result)
#     pre_image_path=pngf

def ocr_rec(img_path,times):
    """
    当前一个字是一张图片，CTC识别单字准确率低
    img_path 图片的路径
    times 一个图片水平复制的次数
    """
    current_img=cv.imread(img_path)
    hc,wc,_=current_img.shape
    newImg=np.zeros((hc,wc*times,3),dtype=current_img.dtype)
    for i in range(times):
        newImg[:,wc*i:wc*(i+1),]=current_img.copy()
    ocr_result=ocr.ocr(newImg,cls=False)
    return ocr_result 


def pipline01():
    """
把所有的切割的字的图片都切割，然后拼接n倍，让ocr去识别，识别的结果一样，就是认为识别的正确。
    """
    TIMES=4
    def analysis_ocrdata():
        """
        分析ocr的结果，如果数据都是一样的那就没有问题，这个就是我们要的数据。
        """
        if len(ocr_result[0])==0:
            return {
                "Flag":False,
            }
        text,prob=ocr_result[0][0][1][0],ocr_result[0][0][1][1]
        if len(text)==TIMES:
            xset=set(text)
            if len(xset)==1:
                return {
                    "Flag":True,
                    "word":text[0],
                    "prob":prob
                }
        return  {
                "Flag":False,
            }

    with open( "ocr_rec.text",'w' ) as tmpfile:
        for pngf in tqdm(glob.glob(os.path.join(PROJECT_DIR,'tmp','*word*','**','*.png'),recursive=True )):
            ocr_result=ocr_rec(pngf,TIMES)
            ret=analysis_ocrdata()
            if ret["Flag"]:
                tmpfile.write("{}\t{}\t{}\n".format(ret["word"],ret["prob"],pngf))
                




    

if __name__=="__main__":
    pipline01()
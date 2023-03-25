import os
PROJECT_DIR= os.path.dirname(
    os.path.dirname(os.path.realpath( __file__))
)
import cv2 as  cv
from tqdm import tqdm
from paddleocr import PaddleOCR
import glob
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import shutil


def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
    """
    按照字体画字，然后等比例缩放到一个canvas_size*canvas_size的图片上。
    """
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    text_region_img=img.crop((l,u,r,d))
    #text_region_img.show()
    text_high, text_wide = text_region_img.size
    new_len_side=max(text_high,text_wide)
    square_img=Image.new("L",(new_len_side,new_len_side),0)
    square_img.paste(text_region_img,( (new_len_side-text_high)//2, (new_len_side-text_wide)//2))
    square_img = square_img.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    #square_img.show()
    np_square_img=255-np.array(square_img)
    square_img=Image.fromarray(np_square_img)
    #square_img.show()
    return square_img


def draw_img2font_example(ch, src_image_path, dst_font, canvas_size, x_offset, y_offset):
    """
    把字体的生成的图片和原始的图片拼接起来
    """
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    src_image=cv.imread(src_image_path,cv.IMREAD_GRAYSCALE)#直接读灰度
    
    example_img = Image.new("L", (canvas_size * 2, canvas_size), 255)# 原来采用RGB格式，然后再转回到L格式，不知道为什么？
    example_img.paste(dst_img, (0, 0))
    example_img.paste(Image.fromarray(src_image), (canvas_size, 0))
    return example_img


def pipeline01():
    """
    tmp/word2imgtop10_64中的图片和生成标准的图片拼接到一块
    """
    SRC_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","word2imgtop10_64")
    DST_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","simple")
    os.makedirs(DST_IMG_DIR,exist_ok=True)
    FONT_PATH="/home/liukun/gan/zi2zi-paddle/data/font/方正楷体_GBK.ttf"
    char_size=55
    dst_font = ImageFont.truetype(FONT_PATH, size=char_size)
    Canvas_Size=64
    sample_dir=""
    count=0
    for word_dir_name in tqdm(os.listdir(SRC_IMG_DIR)):
        word=word_dir_name.split("@")[-1]
        if len(word)>1:
            continue 
        word_dir_path=os.path.join(SRC_IMG_DIR,word_dir_name)
        for image_file in os.listdir(word_dir_path):
            image_file_path=os.path.join(word_dir_path,image_file)
            merge_image=draw_img2font_example(word,image_file_path,dst_font=dst_font,canvas_size=Canvas_Size,x_offset=0,y_offset=0)
            if merge_image:
                merge_image.save(
                    os.path.join(
                        DST_IMG_DIR,"0_%05d.jpg"%(count,)
                    )
                )
            count+=1
if __name__=="__main__":
    #pipline01()
    #countPicWidth() 
    #countWordFreq()
    #display_word()
    #WordImgSet("/home/liukun/ocr/DocumentAI_OCR/tmp/validWordImgs/安")
    #pipline02()
    pipeline01()
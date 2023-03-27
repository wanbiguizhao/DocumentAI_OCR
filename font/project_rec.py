"""
基于投影的方法，识别汉字
"""
from typing import Tuple
import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
import numpy as np  
import os 
import glob
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
PROJECT_DIR= os.path.dirname(
    os.path.dirname(os.path.realpath( __file__))
)
def hproject(image_bin):
    """
    水平投影，识别出行信息
    """
    h,w=image_bin.shape #返回高和宽
    hprojection=np.zeros(image_bin.shape, dtype=np.uint8)
    h_h = [0]*h
    for j in range(h):
        for i in range(w):
            if image_bin[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255

    #cv.imshow("hprojection", hprojection)
    # plt.imshow(hprojection)
    # plt.show()
    return h_h
def slide_window_beg_eng(h_w):
    """
    滑动窗口计算每行汉字的上下边界
    """
    start = 0
    x_start, x_end = [], []
    position = []

    # 根据水平投影获取垂直分割
    for i in range(len(h_w)):
        if h_w[i] > 0 and start == 0:
            x_start.append(i)
            start = 1
        if h_w[i] ==0 and start == 1:
            x_end.append(i)
            start = 0
    if start==1:
        x_end.append(len(h_w)-1)
    return x_start, x_end

def resize(image,new_shape: Tuple[int, int],padding_color: Tuple[int] = (255, )):
    # 对图像进行缩放，多余的部分填充为白色。
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    new_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=padding_color)
    return cv.resize(new_image,new_shape)
def vProject(image_bin):
    h, w = image_bin.shape
    # 垂直投影，获得每个字的边界
    vprojection = np.zeros(image_bin.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if image_bin[j, i ] == 0:
                w_w[i] += 1

    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255
    #cv2.imshow('vpro', vprojection)
    return w_w

def seg_sentences_image():
    BASE_IMAGE_DIR="tmp/ocrSentences"
    #DST_IMAGE_DIR="tmp/ocrSentences_resize"
    width_list=[]
    for image_path in tqdm(glob.glob(os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,"*","*.png"),recursive=True)[:2000]):
        image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
        # 目录相关操作
        (image_dir_path,image_name)=os.path.split(image_path)
        (image_pure_name,extension)=os.path.splitext(image_name)
        DST_IMAGE_DIR=image_dir_path.replace("ocrSentences","ocrSentences_resize")# 没有办法，硬编码一下吧


        os.makedirs(DST_IMAGE_DIR,exist_ok=True)
        blur = cv.GaussianBlur(image,(5,5),0)
        ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #print(image_dir_path,image_pure_name,extension)
        w_w=vProject(th_image)
        w_start,w_end=slide_window_beg_eng(w_w)
        assert len(w_start)==len(w_end)
        width_list=list(map(lambda x, y: x - y, w_end, w_start))
        width_mean=np.mean(width_list)
        width_var=np.std(width_list)
        #print(width_var,width_mean)
        # 计算一下均值和方差，如果一个字符的宽度，小于
        i=0
        word_position_list=[]
        while i < len(w_start):
            word_width=w_end[i]- w_start[i]
            if word_width==0:
                i=i+1
                continue
            if word_width<width_mean:
                if i+1<len(w_start) and (word_width+w_end[i+1]-w_start[i+1])<(2*width_var+width_mean):
                    word_position_list.append([w_start[i],w_end[i+1]])
                    # 合并单词，跳跃到下一个。
                    i=i+1
            elif word_width>(2*width_var+width_mean):
                # 应该是存在粘连问题
                word_position_list.append([w_start[i],w_start[i]+word_width//2])
                word_position_list.append([w_start[i]+word_width//2,w_end[i]])
            else:
                word_position_list.append([w_start[i],w_end[i]])
            i=i+1
        format_64_image_list=[]# 分辨率是64的像素
        for word_position in word_position_list:
            start,end=word_position
            word_image=image[:,start:end]
            image_64=resize(word_image,[64,64])
            format_64_image_list.append(image_64)
        merge_image=np.concatenate(format_64_image_list,axis=1)# 水平方向合并
        plt.imsave(os.path.join(DST_IMAGE_DIR,image_pure_name+extension), merge_image,cmap=plt.gray() )
        # plt.imshow(merge_image)
        # plt.show()
        # print(word_position_list)

def seg_row_image(num_cpu=1):
    # 把一篇文章切割到行
    BASE_IMAGE_DIR="tmp/images"
    DST_IMAGE_DIR="tmp/project_ocrSentences"
    def proc_image(image_path):
        # 切割出来的行，存储起来
        image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
        # 目录相关操作
        (image_dir_path,image_name)=os.path.split(image_path)
        (image_pure_name,extension)=os.path.splitext(image_name)

        DST_IMAGE_DIR=image_dir_path.replace("images","project_ocrSentences")# 没有办法，硬编码一下吧
        os.makedirs(DST_IMAGE_DIR,exist_ok=True)
        blur = cv.GaussianBlur(image,(5,5),0)
        ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #print(image_dir_path,image_pure_name,extension)
        h_h=hproject(th_image)
        h_start,h_end=slide_window_beg_eng(h_h)
        assert len(h_start)==len(h_end)
        height_list=list(map(lambda x, y: x - y, h_end,h_start ))
        height_mean=np.mean(height_list)
        #height_std=np.std(height_list)
        i=0
        while i < len(h_start):
            height_row=h_end[i]- h_start[i]
            cropImg = image[max(0,h_start[i]-1): min(image.shape[0],h_end[i]+1), :]
            i+=1
            if height_row<height_mean :
                continue
            cv.imwrite(
               os.path.join( DST_IMAGE_DIR,image_pure_name+'_'+str(i).rjust(3,"0")+extension )
                ,cropImg)  


    
    if  num_cpu==1:
        for image_path in tqdm(glob.glob(os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,"*","*.png"),recursive=True)[:100]):
            proc_image(image_path)
    else:
        pool=Pool(processes=num_cpu)
        pool.map(proc_image, 
                glob.glob(os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,"*","*.png"),recursive=True) )
        pool.join()
        pool.close()
            
    
def cut_word(image,image_name):
    #把一张图切割成若干个包含汉字的图片
    print(image_name)
    h_h=hproject(image)
    h_start, h_end =slide_window_beg_eng(h_h)
    # 先切割出行来
    position = []
    for i in range(len(h_start)):
        if h_end[i]-h_start[i]<20:
            #print(i,h_end[i]-h_start[i])
            # 这里是一个可能是干扰的
            continue
        cropImg = image[h_start[i]:h_end[i], :]
        plt.imshow(cropImg)
        continue
        w_w=vProject(cropImg)
        
        # w_start=[]
        # w_end=[]
        w_start , w_end = slide_window_beg_eng(w_w)
        # for j in range(len(w_w)):
        #     if w_w[j] > 0 and wstart == 0:
        #         w_start.append(j)
        #         wstart = 1

        #     if w_w[j] ==0 and wstart == 1:
        #         w_end.append(j)
        #         wstart = 0
        # if wstart==1:
        #     w_end.append(len(w_w)-1)
        wstart,wend=0,0
        for k in range(len(w_start)):
            # 单个汉字可能出现粘连的情况，或者单个汉字被劈开的情况。
            # 应该先计算一下平局值，如果，连续的两个字符都是小于平均宽度，并且，两个加起来不超过平均宽度的120%，那么就可以认为可以组合到一起。
            wstart,wend=w_start[k],w_end[k]
            if k>0 and wstart-w_end[k-1]<5 and wend-wstart <20 :
                wstart=w_start[k-1]
            position.append([wstart, h_start[i], wend, h_end[i]])
            wend = 0
    i=0
    num=6
    word_images=[]
    show=True
    for p in position:
        #plt.subplot(num,num,i%(num**2)+1)
        #print(p)
        #plt.imshow(image[p[1]:p[3],p[0]:p[2]],cmap=plt.gray(),aspect='auto')#灰度图正确的表示方法
        word_images.append(image[p[1]:p[3],p[0]:p[2]])
        if i%(num**2)==num**2-1 and show :
            #plt.show()
            show=False
        i+=1
    return word_images

def main():
    def save_image(image_path,image):
        cv.imwrite( image_path ,image) 
    # IMAGE_PATH="/home/aistudio/work/images/1954-01/1954-01_03.png"
    # DST_PATH="/home/aistudio/work/words"
    # image=cv.imread(IMAGE_PATH,cv.IMREAD_GRAYSCALE)
    # img=image
    # # global thresholding
    # ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # # Otsu's thresholding
    # ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv.GaussianBlur(img,(5,5),0)
    # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    BASE_IMAGE_DIR="tmp/images"
    DST_IMAGE_DIR="/home/aistudio/work/word_images"
    index=0
    range_beg=0
    step=1000
    SAVE_DIR="%07d-%07d"%(range_beg,range_beg+step)
    #os.makedirs(os.path.join(PROJECT_DIR,DST_IMAGE_DIR,SAVE_DIR),exist_ok=True)
    for image_path in glob.glob(os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,"*","*.png"),recursive=True):
        image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
        (image_dir_path,image_name)=os.path.split(image_path)
        (image_pure_name,extension)=os.path.splitext(image_name)
        blur = cv.GaussianBlur(image,(5,5),0)
        ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cut_word(th_image,image_pure_name)



        # IMAGE_DIR=os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,image_dir)
        # print(IMAGE_DIR)
        # if not os.path.isdir(IMAGE_DIR):
        #     continue
        # print(IMAGE_DIR)
        # for image_name in os.listdir(IMAGE_DIR):
        #     IMAGE_PATH=os.path.join(IMAGE_DIR,image_name)
        #     image=cv.imread(IMAGE_PATH,cv.IMREAD_GRAYSCALE)
        #     blur = cv.GaussianBlur(image,(5,5),0)
        #     ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #     word_images=cut_word(th_image,image_name)
        #     for img in word_images:
        #         if img is None:
        #             continue
        #         if index>=range_beg+step:
        #             range_beg+=step
        #             SAVE_DIR="%07d-%07d"%(range_beg,range_beg+step)
        #             os.makedirs(os.path.join(DST_IMAGE_DIR,SAVE_DIR),exist_ok=True)
        #             print(SAVE_DIR)
        #         image_path=os.path.join(DST_IMAGE_DIR,SAVE_DIR,str(index).rjust(7,'0')+".png")
        #         #print(image_path)
        #         save_image(image_path,img)
        #         index+=1

if __name__=="__main__":
    seg_sentences_image()
    #seg_row_image(num_cpu=6)
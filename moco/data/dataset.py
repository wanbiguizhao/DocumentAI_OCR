import os
import time
import cv2
import numpy as np
from paddle.io import Dataset
from glob import glob
from tqdm import tqdm
import cv2 as cv 


def imageStrip(image_bin):
    # 切除行中两边白色的地方
    h, w = image_bin.shape
    # 垂直投影，获得每个字的边界
    vprojection = np.zeros(image_bin.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        #w_w[i]=sum(image_bin[:,i])
        for j in range(h):
            if image_bin[j, i ] == 0:
                w_w[i] += 1
    beg_index=0
    while beg_index<len(w_w):
        if w_w[beg_index]<=5:
            beg_index+=1
        else:
            break
    end_index=w -1
    while end_index>beg_index:
        if w_w[end_index]<=5:
            end_index-=1
        else:
            break
    return beg_index-16,end_index+16
def splitImage(image_bin):
    """
    将图片切割成16*48的小片段，步长为4
    """
    # 第一步假设高度已经填充为48吧，宽度是4的整数倍
    step=8
    h,w=image_bin.shape
    beg_index=0
    end_index=15
    location_list=[]
    while end_index<w:
        location_list.append((beg_index,end_index))
        beg_index+=step 
        end_index+=step
    return location_list
        



class WordImagePiceDataset(Dataset):
    """
    数据集，可以自行将 图片切割成 16*48的数据集
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, data_dir, label_path=None, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(WordImagePiceDataset, self).__init__()
        self.data_list = []
        for image_path in tqdm(glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:20]):
            image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
            # 切掉白色的两边
            blur = cv.GaussianBlur(image,(5,5),0)
            ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            h,w=th_image.shape
            if h>=48:
                continue 
            beg_index,end_index=imageStrip(th_image)
            w=end_index-beg_index+1
            h_padding=48-h
            w_padding=(4-w%4)%4
            top, bottom = h_padding//2, h_padding-(h_padding//2)# 上下部分填充
            left,right=w_padding//2,w_padding-(w_padding//2)
            new_image = cv.copyMakeBorder(image[:,beg_index:end_index+1], top, bottom, left, right, cv.BORDER_CONSTANT, value=(255,))
            self.data_list.extend([  new_image[: ,beg:end+1 ] for beg,end in splitImage(new_image) ])
            #print(image_dir_path,image_pure_name,extension)
            #填充到48的整数倍 
        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image = self.data_list[index]
        image=image.astype('float32')
        # 读取灰度图
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # image = image.astype('float32')
        # # 应用数据处理方法到图像上
        # if self.transform is not None:
        #     image = self.transform(image)
        # # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        # #label = int(label)
        # 返回图像和对应标签
        return image, 0

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


if __name__=="__main__":
    from matplotlib import pyplot as plt
    ds=WordImagePiceDataset(data_dir="tmp/project_ocrSentences")
    print(len(ds))
    plt.figure(32)
    for x in range(32):
        plt.subplot(1,32,x+1)
        plt.imshow(ds[x][0])
    plt.show()
    time.sleep(10) 

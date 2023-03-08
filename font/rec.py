"""
调用paddle，为每个字找打对应的图片
"""
import os
import tempfile 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(os.path.realpath( __file__))
)
import cv2 as  cv
from tqdm import tqdm
from paddleocr import PaddleOCR
import glob
from PIL import Image
import numpy as np
import shutil
# ocr = PaddleOCR(
#     det=False,
#     rec=True,
#     lang="chinese_cht",
#     cls=False
# )  
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
# 统计一下word图片的宽度，统计一下。                
from matplotlib import pyplot as plt 
from collections import Counter 
from collections import defaultdict
def countPicWidth():
    """
    统计每个图片宽度
    """
    width_counter=Counter()
    higth_counter=Counter()
    ratio_counter=Counter()
    # 两个字的宽度大概在 88,87,89 大概是3w个图片
    for word_png_path in tqdm(glob.glob(os.path.join(PROJECT_DIR,'tmp','*word*','**','*.png'),recursive=True )):
        current_img=cv.imread(word_png_path)
        hc,wc,_=current_img.shape
        width_counter.update([wc])
        higth_counter.update([hc])
        ratio_counter.update([round(wc/hc,1)])
    print(width_counter.most_common(100))
    print(higth_counter.most_common(100))
    print(ratio_counter.most_common(100))

def countWordFreq():
    """
    统计汉字出现的频率
    """
    import shutil
    wordCounter=Counter()
    with open(os.path.join(PROJECT_DIR, 'sort_ocr_rec.text'),'r') as sst:
        alldata=sst.readlines()
        for ol in tqdm(alldata):
            da=ol.split("\t")
            if len(da)==3:
                wordCounter.update([da[0]])
                # 按照字，创建文件夹，然后把ocr识别的图片放到对应的目录中
                # png_dir=os.path.join(PROJECT_DIR,'tmp','wordmappings',da[0])
                # if not os.path.exists(png_dir):
                #     os.makedirs(png_dir)
                # abs_pn_path=da[2].replace('\n','')
                # shutil.copy(abs_pn_path,png_dir)
        with open(os.path.join(PROJECT_DIR, 'sort_word_freq.text') ,'w') as swf:
            for f,w in sorted([ (f,w) for w,f in wordCounter.items()],reverse=True):
                swf.write(
                    "{}-{}\n".format(w,f)
                )
def get_topn_word():
    """
    从sort_ocr_rec.text中拿到每个汉字概率最高的10个图片
    """
    BASE_WORD_IMAGE_DST=os.path.join(PROJECT_DIR,'tmp',"word2imgtop10")
    if not os.path.exists(BASE_WORD_IMAGE_DST):
        os.makedirs(BASE_WORD_IMAGE_DST)
    from collections import defaultdict
    word_index_range_dict=defaultdict(list)
    with open(os.path.join(PROJECT_DIR, 'ocr_rec.text'),'r') as sst:
        alldata=sst.readlines()
        alldata=sorted(alldata)
        wordCounter=Counter()
        current_word="init"
        word_index_range_dict[current_word].append(-1)
        for index, dataline in tqdm(enumerate(alldata)):
            word_info_list=dataline.split("\t")
            if len(word_info_list)==3:
                if current_word!=word_info_list[0]:
                    word_index_range_dict[current_word].append(index-1)
                    current_word=word_info_list[0]
                    word_index_range_dict[current_word].append(index)
                wordCounter.update([word_info_list[0]])
        word_index_range_dict[current_word].append(index)
        del wordCounter["init"]
        del word_index_range_dict["init"]
        index=0
        for count,word in sorted([ (count,word) for word,count in wordCounter.items()],reverse=True):
            begindex,endindex=word_index_range_dict[word]
            WORD_DST_DIR=os.path.join(BASE_WORD_IMAGE_DST,str(index).rjust(5,'0')+"@"+word)
            if not os.path.exists(WORD_DST_DIR):
                os.makedirs(WORD_DST_DIR)
            for dataline in alldata[max(begindex,endindex+1-10):endindex+1]:
                word_info_list=dataline.replace("\n","").split("\t")
                shutil.copy(word_info_list[2],WORD_DST_DIR)
            index+=1
                




        

def display_word():
    """
    展示一个字的图片
    统计每个字在sort_ocr_rec.text 中的开始位置和结束位置
    """
    word_index_range=defaultdict(lambda:[])
    with open(os.path.join(PROJECT_DIR, 'sort_ocr_rec.text'),'r') as sst:
        alldata=sst.readlines()
        beg_index=0
        pre_word=alldata[0][0]
        for index, ol in enumerate(alldata):
            da=ol.split("\t")
            if len(da)==3:
                if not pre_word==da[0]:
                    word_index_range[pre_word]=[beg_index,index-1]
                    pre_word=da[0]
                    beg_index=index 
        word_index_range[pre_word]=[beg_index,index-1]
        with open(os.path.join(PROJECT_DIR, 'sort_word_freq.text') ,'r') as swf:
            line=swf.readline()
            plt.figure()

            while line:
                word,freq=line.split('-')
                beg_index,end_index=word_index_range[word]
                for index,img_info in enumerate(alldata[beg_index:end_index]):
                    if index<8*8:
                        plt.subplot(8,8,index+1)
                        word_img=cv.imread(img_info.split('\t')[-1].replace("\n",""))
                        plt.imshow(word_img)
                    else:
                        break 
                plt.show(block=True)
                    
                    
                line=swf.readline()
    print(word_index_range)
             


import random
class WordImgSet:
    """
    字对应的字符
    """
    def __init__(self,word_dir) -> None:
        for img_name in  os.listdir(word_dir):
            self.word=self.get_word(word_dir)
            self.image_lists=[]
            self.index=-1
            #print(img_name)
            img_path=os.path.join(word_dir,img_name)

            img=cv.imread(img_path)
            newimg=cv.resize(img,(45,47)) # 不再对原来的代码进行resize
            #统一对字符图片的大小做归一化。
            #print(newimg.shape,img.shape)
            self.image_lists.append(newimg)
            #cv.imshow("new img",newimg)
            #cv.waitKey(0)
            #cv2.destroyAllWindows()
    
    def get_one_img(self):
        index=random.randint(0,len(self.image_lists)-1)
        #self.index=self.index%len(self.image_lists)
        return self.image_lists[index]
    def get_word(self,word_dir_name):
        return word_dir_name[-1]

class WordImgSetFomat01(WordImgSet):
    """
    word_dir name formate is 123123@wordword
    """

    def get_word(self, word_dir_name):

        return word_dir_name.split("@")[-1]

class BuildSentencCorups:

    def __init__(self,wordCount=10,sentence_num=10000,random_sentence_len=True,WordImgSetClass=WordImgSet
                            ,BASE_WORD_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","validWordImgs")
                            ,BASE_FASKE_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","output")
                            ) -> None:
        self.wordCount=wordCount
        self.sentence_num=sentence_num
        self.random_sentence_len=random_sentence_len
        self.WordImgSetClass=WordImgSet
        self.BASE_WORD_IMG_DIR=BASE_WORD_IMG_DIR
        self.BASE_FASKE_IMG_DIR=BASE_FASKE_IMG_DIR

    def getRandomSentence(self):
        length=self.sentence_num
        wordCount=self.wordCount
        allHanWords=[]
        BASE_WORD_IMG_DIR=self.BASE_WORD_IMG_DIR
        for word_dir_name in os.listdir(BASE_WORD_IMG_DIR):
            word=self.WordImgSetClass(os.path.join(BASE_WORD_IMG_DIR,word_dir_name))
            allHanWords.append(word)
        #print([w.word for w in allHanWords])
        random_sentence=[]
        for th in range(length):
            sentence_data=[]
            for _ in range(random.randrange(5,wordCount)):
                sentence_data.append(
                    allHanWords[ random.randrange(len(allHanWords)) ]
                )
            print(th, "".join( [ w.word for w in sentence_data]))
            random_sentence.append(sentence_data)
        return random_sentence
    def buildSentenceImg(self,random_sentence):
        """
        根据校验的汉字，生成ocr的训练数据
        """

        BASE_FASKE_IMG_DIR=self.BASE_FASKE_IMG_DIR
        IMGS_DIR=os.path.join(BASE_FASKE_IMG_DIR,"images")
        labels_data=[]
        for index,onesentence in enumerate(random_sentence):
            first_word_img=onesentence[0].get_one_img()
            h,w,_=first_word_img.shape
            sentence_img=np.zeros((h,w*len(onesentence),3),dtype=first_word_img.dtype)
            sentence_img[:,:,:]=255
            beg_w=0
            label=""
            for wordobj in  onesentence:
                wordimg=wordobj.get_one_img()
                sentence_img[:,beg_w:beg_w+w,:]=wordimg.copy()
                beg_w+=w 
                label+=wordobj.word
            png_name="{}".format(index).rjust(6,"0")+".png"
            cv.imwrite( os.path.join(IMGS_DIR ,png_name),sentence_img)
            labels_data.append("images\{}\t{}\n".format(png_name,label))
        with open(
            os.path.join(BASE_FASKE_IMG_DIR,"labels.text"),"w"
        ) as lf:
            lf.writelines(labels_data)
    
    def build(self):
        random_sentence=self.getRandomSentence()
        self.buildSentenceImg(
            random_sentence
        )



def getRandomSentence(length=100000,wordCount=12):
    """
    随机生成若干个句子
    """
    allHanWords=[]
    BASE_WORD_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","validWordImgs")
    for word_dir_name in os.listdir(BASE_WORD_IMG_DIR):
        word=WordImgSet(os.path.join(BASE_WORD_IMG_DIR,word_dir_name))
        allHanWords.append(word)
    #print([w.word for w in allHanWords])
    random_sentence=[]
    for th in range(length):
        sentence_data=[]
        for _ in range(random.randrange(5,wordCount)):
            sentence_data.append(
                allHanWords[ random.randrange(len(allHanWords)) ]
            )
        print(th, "".join( [ w.word for w in sentence_data]))
        random_sentence.append(sentence_data)
    return random_sentence
    
 
def buildSentenceImg(random_sentence):
    """
    根据校验的汉字，生成ocr的训练数据
    """
    BASE_FASKE_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","output")
    labels_data=[]
    for index,onesentence in enumerate(random_sentence):
        first_word_img=onesentence[0].get_one_img()
        h,w,_=first_word_img.shape
        sentence_img=np.zeros((h,w*len(onesentence),3),dtype=first_word_img.dtype)
        sentence_img[:,:,:]=255
        beg_w=0
        label=""
        for wordobj in  onesentence:
            wordimg=wordobj.get_one_img()
            sentence_img[:,beg_w:beg_w+w,:]=wordimg.copy()
            beg_w+=w 
            label+=wordobj.word
        png_name="{}".format(index).rjust(6,"0")+".png"
        cv.imwrite(os.path.join(BASE_FASKE_IMG_DIR,"images" ,png_name),sentence_img)
        labels_data.append("image\{}\t{}\n".format(png_name,label))
    with open(
        os.path.join(BASE_FASKE_IMG_DIR,"labels.text"),"w"
    ) as lf:
        lf.writelines(labels_data)

        
def pipline02():
    """
    根据word对应的图片，自动装生成标注数据，用于paddle的训练
    """
    random_sentences=getRandomSentence(length=50000,wordCount=10) 
    buildSentenceImg(random_sentences)



def pipline03():
    """
    处理格式为
    ├── 01974@匈
    │   └── 380349.png
    ├── 01975@匆
    │   └── 092312.png
    ├── 01976@勾
    │   └── 072758.png
    的文件
    """
    def check_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path,exist_ok=True)
        

    BASE_WORD_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","word2imgtop10")
    BASE_FASKE_IMG_DIR=os.path.join(PROJECT_DIR,"tmp","output")
    FAKE_IMGS_DIR=os.path.join(BASE_FASKE_IMG_DIR,"images")
    check_dir(BASE_FASKE_IMG_DIR)
    check_dir(BASE_WORD_IMG_DIR)
    check_dir(FAKE_IMGS_DIR)
    
    build=BuildSentencCorups(wordCount=12,sentence_num=500,WordImgSetClass=WordImgSetFomat01,
            BASE_WORD_IMG_DIR=BASE_WORD_IMG_DIR,
            BASE_FASKE_IMG_DIR=BASE_FASKE_IMG_DIR
            )
    build.build()


                
                



         



    

if __name__=="__main__":
    #pipline01()
    #countPicWidth() 
    #countWordFreq()
    #display_word()
    #WordImgSet("/home/liukun/ocr/DocumentAI_OCR/tmp/validWordImgs/安")
    #pipline02()
    pipline03()
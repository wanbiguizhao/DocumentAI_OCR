import os 
import random
# 用于对数据分割的工具。
from sklearn.model_selection import train_test_split
PROJECT_DIR= os.path.dirname(__file__)
DATA_DIR=os.path.join(PROJECT_DIR,'Data')
def merge_labels_info(base_dir,save_dir=None):
    """
    遍历目录下所有的文件文件夹，找到包含labels.txt 和png图片
    并且生成，相对于base_dir,下所有图片的相对位置信息。
    """
    rec_labels_info=[]
    for xdir in os.listdir(base_dir):
        if not os.path.isdir( os.path.join(base_dir,xdir)):
            continue 
        labels_file_path=os.path.join(base_dir,xdir,'labels.txt')
        assert os.path.exists(labels_file_path)# 不存在数据就报错，需要严格保证数据的文件组织形式
        with open(labels_file_path,'r') as lablesf:
            for line in lablesf.readlines():
                xd=line.split('\t')
                if len(xd)!=3:
                    continue
                rel_path,img_str,rec_prob=xd[0],xd[1],xd[2]
                rec_labels_info.append(
                    "{}\t{}\t{}".format(os.path.join(xdir,rel_path),img_str,rec_prob)
                )
    print("the length rec_labels_info :  ",len(rec_labels_info))
    if save_dir:
        os.makedirs(save_dir,exist_ok=True)
        with open( os.path.join(save_dir,'merge_labels.txt'),'w') as mlablesf:
            mlablesf.writelines(rec_labels_info)
def countFreq(file_path=os.path.join(PROJECT_DIR,"tmp",'val.txt')):
    from collections import Counter
    # 临时使用统计val中，每个汉字的使用总数。
    val_word_counter=Counter()
    with open(file_path,"r") as labelf:
        for line in labelf.readlines():
            xd=line.split('\t')
            if len(xd)!=3:
                continue
            val_word_counter.update(xd[1])
    for ele in sorted([ (f,w) for (w,f) in val_word_counter.items()],reverse=True):
        print(ele[1],'\t',ele[0])



def make_train_val(labels_dir,labels_name="merge_labels.txt",train_per=0.7):
    #在 labels_dir目录下生成train 和 val 数据
    assert os.path.exists(labels_dir)
    label_path=os.path.join( labels_dir,labels_name) 
    assert os.path.exists(label_path )
    with open(label_path,"r") as labelf:
        labels_data=labelf.readlines()
        train,val=train_test_split(labels_data,shuffle=True,train_size=train_per,random_state=99)
        print(train,val)
        with open( os.path.join(labels_dir,'train.txt'),'w' ) as tf:
            tf.writelines(train)
        with open(os.path.join(labels_dir,'val.txt'),'w') as vf:
            vf.writelines(val) 
    


if __name__=="__main__":
    #merge_labels_info(DATA_DIR,save_dir=os.path.join(PROJECT_DIR,'tmp'))
    #make_train_val(os.path.join(PROJECT_DIR,'tmp','output'),labels_name="labels.text")
    countFreq()
    os.path.isdir()
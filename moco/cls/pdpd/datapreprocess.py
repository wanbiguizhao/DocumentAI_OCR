
# %%
import os 
PROJECT_DIR= os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.realpath( __file__))
                                    )
                                )
                            )
DATASET_DIR=os.path.join(PROJECT_DIR,"moco","dataset")
from sklearn.model_selection import train_test_split 
import glob

def explain_labels():
    pass 
def load_dataset(dataset_dir):
    import_data_index=[]# 
    WORD_TYPE=0# 表示图像是汉字的一部分
    SPACE_TYPE=1# 表示图像是两个汉字中间间隔
    SPACE_TYPE_LIST=[]
    for dir_path in os.listdir(dataset_dir):
        ds_image_dir=os.path.join(dataset_dir,dir_path)
        #print(ds_image_dir)
        labels_path=os.path.join(dataset_dir,dir_path,'labels.txt')
        if not os.path.exists(labels_path) or not os.path.isfile(labels_path):
            # 文件不存在
            assert False 
        dataset_image_list=sorted(glob.glob(os.path.join(ds_image_dir,"word*.png")))
        #print(dataset_image_list)
        with open(labels_path,'r') as lab_file:
            # 找到目录下对应的
            for rowdata in lab_file.readlines():
                #print(rowdata)
                clean_row=rowdata.strip("\n").strip("\t")
                if clean_row:
                    #print(clean_row)
                    if "-" in clean_row:
                        #218-224 表示label的某个区间都是SPACE_TYPE
            # 返回的应该是[dataset_dir下的路径，标签]


            

            
#%%
load_dataset(DATASET_DIR)    

# %%

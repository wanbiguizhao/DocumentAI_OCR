
# %%
import os
import re 
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

def explain_labels(clean_row,labels_path):
    """
    对
    35+*
    35++
    35-40*
    35+3
    这一类的数据进行解析
    """
    import_flag=False 
    # 写一个状态机吧
    num_list=[]
    find_num_flag=False
    current_num=-1
    # 要求使用两个堆栈
    # 一个是数字堆栈
    # 一个是符号堆栈
    num_stack=[]
    flag_stack=[]
    for x in clean_row:
        if x.isdigit():
            if not find_num_flag:
                find_num_flag=True
                current_num=int(x) 
            else:
                current_num=current_num*10+int(x)
        else:
            if find_num_flag:
                num_stack.append(current_num)
                find_num_flag=False
            if   x not in "*+-":
                print(labels_path,'->',clean_row,"Data Format Error ")
                raise Exception()
            flag_stack.append(x)
            
    
    if find_num_flag:
        num_stack.append(current_num)
    assert len(num_stack)!=0
    # 开始对数据进行处理
    while len(flag_stack)>0:
        if flag_stack[-1]=="*":
            import_flag=True
            flag_stack.pop(-1)
        elif flag_stack[-1]=="+":
            plus_count=0
            while flag_stack and flag_stack[-1]=="+":
                flag_stack.pop(-1)
                plus_count+=1
            # 此时flag_stack必须清空了
            assert len(flag_stack)==0 
            if len(num_stack)==1:
                # 35++++ 这种情况
                num_list= list(range(num_stack[0],num_stack[0]+plus_count+1))
            elif len(num_stack)==2:
                # 35+3 的情况
                # 不能出现 35++3的情况
                assert plus_count==1
                num_list=list(range(num_stack[0],num_stack[1]+num_stack[0]+1 ))
            else:
                # 数字太多了
                print(labels_path,'->',clean_row,"Data Format Error ")
                raise Exception()
        elif flag_stack[-1]=="-":
            assert len(num_stack)==2 and len(flag_stack)==1
            flag_stack.pop(-1)
            num_list=list(range(num_stack[0],num_stack[1]+1))
        else:
            print(labels_path,'->',clean_row,"Data Format Error ")
            raise Exception()
    # 没有出现过特殊符号的情况
    if len(flag_stack)==0 and len(num_list)==0:
        num_list=num_stack
    print(clean_row,"\t",num_list,"\t",import_flag) 
def do_image_name_index_check(dataset_image_list):
    """
    对数据进行检查，确保图片的名称和索引可以一一对应上，
    例如：word_seg_00001_type_05.png  
        00001 
        代表dataset_image_list[1]=="word_seg_00001_type_05.png"
    """
    for index, image_info in enumerate(dataset_image_list):
        matchObj = re.match(
        r'.*word.*_(?P<id_ds>\d+)_.*_(?P<id_cluster>\d+)', image_info)
        assert matchObj
        id_ds=matchObj.groupdict()["id_ds"]
        assert index==int(id_ds)
     
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
        do_image_name_index_check(dataset_image_list)
    
        with open(labels_path,'r') as lab_file:
            # 找到目录下对应的
            
            for rowdata in lab_file.readlines():
                #print(rowdata)
                clean_row=rowdata.strip("\n").strip("\t")
                if not clean_row:
                    continue 
                    #print(clean_row)



            # 返回的应该是[dataset_dir下的路径，标签]


            

            
#%%
load_dataset(DATASET_DIR)    

# %%

from collections import defaultdict
import os 

import os
PROJECT_DIR=  os.path.dirname(os.path.realpath( __file__))


def load_data():
    han_dict=defaultdict(list)
    with open("image_top_score_info.txt","r") as df:
       for rowdata in df.read().splitlines(): 
            if len(rowdata)<10:
               continue
            data_list=rowdata.split("\t")
            han_dict[data_list[0]].append(
               {
                   "score":data_list[1],
                   "image_uuid":data_list[2],
                   "image_path":data_list[3],
                   "han":data_list[0]
               }
            )
    return han_dict
    #print(han_dict)
def render_html():
    from jinja2 import Environment,FileSystemLoader, PackageLoader, select_autoescape
    import re 
    env = Environment(
        loader=FileSystemLoader(PROJECT_DIR),
        autoescape=select_autoescape()
    )
    template = env.get_template("ocrclient/template/image_han_talbe.html")
    table_list=[]# 存儲多個表
    han_dict=load_data()
    table_col_num=40# 一個表有20列

    key_list=sorted(han_dict.keys(),key=lambda x: -len(han_dict[x]))# 做了排序，可以保證先看到多的圖片

    beg=0
    while beg+table_col_num<len(key_list):
        th_list=key_list[beg:beg+table_col_num]
        table_data=[]
        row_index=0
        while row_index<10:
            if row_index>=len(han_dict[th_list[0]]):#一個漢字對應的圖片個數，因爲是長度進行了排序，所以第一列就是最大長度
                break
            table_row=[]
            for ci in range(table_col_num):
                han_key=th_list[ci]
                han_data=han_dict[han_key]
                if row_index>=len(han_data):
                    break
                table_row.append(
                    han_data[row_index]
                )
            table_data.append(table_row)
            row_index+=1
        if beg>len(key_list)-1000:
            table_list.append(
                {
                    "th_list":th_list,
                    "table_data":table_data,
                }
            )
        beg+=table_col_num
        

    html_data=template.render(table_list=table_list)
    with open("hanimagetable.html",'w') as ith:
        ith.write(html_data)


if __name__=="__main__":
    #load_data()
    render_html()

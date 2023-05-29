from flask import Flask, render_template, request
from collections import defaultdict
import os 
PROJECT_DIR=  os.path.dirname(os.path.dirname(os.path.realpath( __file__)))
template_dir = os.path.join(PROJECT_DIR,"ocrclient/template")
app = Flask(__name__, template_folder=template_dir)

from flask import send_from_directory
@app.route('/tmp/<path:path>')
def send_tmp_report(path):
    return send_from_directory(f"{PROJECT_DIR}/tmp", path)
@app.route('/easyocr/<path:path>')
def send_easyocr_report(path):
    return send_from_directory(f"{PROJECT_DIR}/easyocr", path)
@app.route('/static/<path:path>')
def send_static_report(path):
    return send_from_directory(f"{PROJECT_DIR}/ocrclient/static", path)
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
@app.route('/',methods = ['POST', 'GET'])
@app.route('/hello',methods = ['POST', 'GET'])
def render_han_image():
#     return render_template('index.html', user=user)
# def render_han_image():
    # from jinja2 import Environment,FileSystemLoader, PackageLoader, select_autoescape
    # import re 
    # env = Environment(
    #     loader=FileSystemLoader(PROJECT_DIR),
    #     autoescape=select_autoescape()
    # )
    # template = env.get_template("ocrclient/template/image_han_talbe.html")
    if request.method=="POST":
        print(request.json)


    table_list=[]# 存儲多個表
    han_dict=load_data()
    table_col_num=40# 一個表有20列

    key_list=sorted(han_dict.keys(),key=lambda x: -len(han_dict[x]))# 做了排序，可以保證先看到多的圖片
    default_value=-1
    beg=0
    han_image_dict=defaultdict(list)
    if beg+table_col_num<len(key_list):
        th_list=key_list[beg:beg+table_col_num]
    else:
        th_list=key_list[beg:]
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
            han_image_dict[han_data[row_index]["image_uuid"]].extend([ han_key,default_value] )# 大致的数据结构是某一张图片，对应的数据结构[汉字，状态] ，状态-1表示错，0对，2是错
            table_row.append(
                han_data[row_index]
            )
        table_data.append(table_row)
        row_index+=1
    table_list.append(
        {
            "han_image_dict":{},
            "th_list":th_list,
            "table_data":table_data,
        }
    )

    return render_template("image_han_talbe.html",han_image_dict=han_image_dict,default_value=default_value,table_list=table_list)



if __name__=="__main__":
    #load_data()
    #render_html()
    app.run(
        debug=True
    )

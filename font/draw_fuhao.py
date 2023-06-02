# 画出来所有的符号

from collections import defaultdict
import os
import shutil
#from util.visualizer import save_single_image
from PIL import Image,ImageDraw,ImageFont
import json
from tqdm import tqdm
def smart_make_dirs(dir_paths):
    if os.path.exists(dir_paths):
        shutil.rmtree(dir_paths)
    os.makedirs(dir_paths)
def draw(font_path,label,save_dir,save_name):
    font = ImageFont.truetype(font_path,55)
    label_w, label_h  = font.getsize(label)
    img_target = Image.new('RGB', (label_w,label_h),(255,255,255))
    drawBrush = ImageDraw.Draw(img_target)
    drawBrush.text((0,0),label,fill=(0,0,0),font = font)
    img_target.save(os.path.join(save_dir,save_name))
    return img_target

def pipline01():
    with open("font/fuhao.txt","r") as fuhao:
        fh_list=fuhao.read().splitlines()
    font_path_list=[]
    font_dir="tmp/font"
    save_dir="tmp/infer_fuhao"
    datadict=defaultdict(list)
    smart_make_dirs(save_dir)
    for font_file in os.listdir(font_dir):
        font_path_list.append(f"{font_dir}/{font_file}")
    for fh_id in range(len(fh_list)):
        for font_id in range(len(font_path_list)):
            draw(font_path_list[font_id],fh_list[fh_id],save_dir,f"{fh_id:03d}-{font_id}.png")
            datadict[fh_list[fh_id]].append(f"{save_dir}/{fh_id:03d}-{font_id}.png")
    with open("ocrclient/tmp/fuhao_image_path.json","w") as fu_file:
        json.dump(datadict,fu_file,indent=2,ensure_ascii=False)


    



if __name__=="__main__":
    #load_corups()
    pipline01()
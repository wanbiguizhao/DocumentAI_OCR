# 把图片上传到easyocr的api，并且下载调用结果。
import requests
import os 
from PIL import Image,ImageDraw
colab_url = os.environ.get("OCR_URL","https://4f23-34-82-27-15.ngrok-free.app/")#'https://0adc-34-143-240-199.ngrok-free.app/'
assert colab_url 
from glob import glob
from tqdm import tqdm
import shutil
import json
from PIL import Image
CURRENT_DATA_DIR=os.path.dirname(os.path.realpath(__file__))
def easyocr_webservice(image_path):
    files = {'file': open(image_path, 'rb')}
    model_results = requests.post(
            colab_url,files=files
        )
    if model_results.status_code==200:
        ocr_result=model_results.json()
        return ocr_result
    return []
def worddetect_webservice(image_bytes):
    files = {'file': image_bytes}
    model_results = requests.post(
            "http://192.168.124.15:8088/",files=files
        )
    if model_results.status_code==200:
        ocr_result=model_results.json()
        return ocr_result
    return []
def get_file_name(file_path):
    _,file_name=os.path.split(file_path) 
    pure_name,ext=os.path.splitext(file_name)
    return pure_name,ext
def smart_mkdir(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)


def pipwork01():
    #调用ocr把图片转化为ocr的数据
    BASE_DATA_DIR=os.path.join(CURRENT_DATA_DIR,"tmp","ocr_data")
    for image_path in tqdm(glob("easyocr/tmp/images/*.png")):
        
        purename,ext=get_file_name(image_path)
        target_data_dir=os.path.join(BASE_DATA_DIR,purename)
        smart_mkdir(target_data_dir)
        shutil.copy2(image_path,f"{target_data_dir}/{purename}{ext}")
        img=Image.open(image_path).convert('L')
        ocr_result_list=easyocr_webservice(image_path)# 水平方向的ocr和竖直方向的ocr
        json_data={
            "ocr":ocr_result_list,
            "image_name":f"{purename}.{ext}",
            "image_path":f"{target_data_dir}/{purename}.{ext}"[len(CURRENT_DATA_DIR)+1:],
            "original_height": img.height,
            "original_width": img.width,
        }
        with open(f"{target_data_dir}/ocr.json",'w')as json_file:
            json.dump(json_data,json_file,indent=True,ensure_ascii=False)
import io

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format="png")
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr
def pipwork02():
    #批量把ocr的数据转换成预定义的convert_ocr格式。
    #把ocr的图片进行切割。
    base_data_dir="easyocr/tmp/ocr_data/"
    for filesdir in tqdm(os.listdir(base_data_dir)):
        seg_hand_data(os.path.join(base_data_dir,filesdir))
def pipwork03():
    #把识别出来的汉字画出来
    base_data_dir="easyocr/tmp/ocr_data/"
    for filesdir in tqdm(os.listdir(base_data_dir)):
        draw_rect(os.path.join(base_data_dir,filesdir))

def draw_rect(data_dir="easyocr/tmp/ocr_data/1954-02_13"):
    print(data_dir)
    smart_mkdir(f"{data_dir}/words")
    smart_mkdir(f"{data_dir}/sentence")
    png_name=data_dir.split("/")[-1]
    draw_img=Image.open(f"{data_dir}/{png_name}.png").convert("RGBA")
    crop_img=Image.open(f"{data_dir}/{png_name}.png").convert("RGBA")
    draw = ImageDraw.Draw(draw_img)
    with open(f"{data_dir}/convert_ocr.json") as json_file:
        ocr_data=json.load(json_file)
    for one_sentece_data in ocr_data["ocr_areas"]:
        sxu,syu=one_sentece_data["x"],one_sentece_data["y"]
        sxd,syd=sxu+one_sentece_data["width"],syu+one_sentece_data["heigh"]
        sentence_crop_image=crop_img.crop([ sxu,syu,sxd,syd])
        sentence_crop_image.save(f"{data_dir}/sentence/{one_sentece_data['id']}.png","png")
        for w in one_sentece_data["words"]:
            xu,yu,xd,yd=w["x"],w["y"],w["x"]+w["width"],w["y"]+w["height"]
            if w["width"]<=30 or w["width"]>=140 or w["height"]>120:
                continue
            word_crop_image=crop_img.crop([ xu-1,yu,xd+1,yd])
            word_crop_image.save(f"{data_dir}/words/{w['id']}.png","png")
            draw.rectangle([(xu,yu),(xd,yd)],outline="yellow")
    draw_img.save(f"{data_dir}/{png_name}_rect.png","png")


    draw.rectangle(((0, 00), (100, 100)), fill="black")
    pass 
def seg_hand_data(data_dir="easyocr/tmp/ocr_data/1954-02_13"):
    png_name=data_dir.split("/")[-1]
    # for image_path in tqdm(glob("easyocr/tmp/ocr_data/*/*.png")):
    #     shutil.move(image_path,image_path.replace('..','.'))
    image=Image.open(f"{data_dir}/{png_name}.png").convert("L")
    # 对image进行切割
    with open(f"{data_dir}/ocr.json") as json_file:
        ocr_data=json.load(json_file)
    new_ocr_data=[]
    for index,one_ocr_data in enumerate(ocr_data["ocr"]):
        four_points,text,score=one_ocr_data[0],one_ocr_data[1],one_ocr_data[2]
        four_points[0],four_points[0]
        crop_image=image.crop([four_points[0][0],four_points[0][1],four_points[2][0],four_points[2][1]])
        crop_image_data={
                "id":f"s-{index+1}",
                "x":four_points[0][0],
                "y":four_points[0][1],
                "width":four_points[2][0]-four_points[0][0],
                "heigh":four_points[2][1]-four_points[0][1],
                "value":{
                        "text":[text],
                        "score":score
                },
                "words":[
            ]
        }  
        crop_image_byte=image_to_byte_array(crop_image)
        word_location_list=worddetect_webservice(crop_image_byte)
        for wi,loc in enumerate(word_location_list):
            crop_image_data["words"].append(
                {
                    "id":f"s-{index+1}_w-{wi+1}",
                    "x":four_points[0][0]+loc[0],
                    "y":four_points[0][1],
                    "width":loc[1]-loc[0],
                    "height":four_points[2][1]-four_points[0][1] 
                }
            )
        new_ocr_data.append(crop_image_data)
        #print("10")
        #
        #建造一个images文件夹，是否把图片都切割出来？
    ocr_data["ocr_areas"]=new_ocr_data
    with open(f"{data_dir}/convert_ocr.json","w") as json_file:
        json.dump(ocr_data,json_file,ensure_ascii=False,indent=2)
       
        
        


def do_ocr(image_path):
    files = {'file': open(image_path, 'rb')}
    model_results = requests.post(
            colab_url,files=files
        )
    h_bbox_list,v_bbox_list=model_results.json()
    return 
    img=Image.open(files['file'])
    for rect in h_bbox_list[0]:
        xu,xd,yu,yd=rect
        crop_img=img.crop([xu,yu,xd,yd])
        crop_img.show()
def do_ocr_det(image_path):
    """
    只识别框
    """
    files = {'file': open(image_path, 'rb')}
    model_results = requests.post(
            colab_url,files=files
        )
    h_bbox_list,v_bbox_list=model_results.json()

def test_do_ocr():
    image_path="tmp/images/1955-12/0.png"
    do_ocr(image_path)

#test_do_ocr()


if __name__=="__main__":
    #pipwork02()
    pipwork03()

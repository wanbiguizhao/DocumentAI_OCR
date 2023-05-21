# 把图片上传到easyocr的api，并且下载调用结果。
import requests
import os 
from PIL import Image
colab_url = os.environ.get("OCR_URL","")#'https://0adc-34-143-240-199.ngrok-free.app/'
assert colab_url 




def do_ocr(image_path):
    files = {'file': open(image_path, 'rb')}
    model_results = requests.post(
            colab_url,files=files
        )
    h_bbox_list,v_bbox_list=model_results.json()
    img=Image.open(files['file'])
    for rect in h_bbox_list[0]:
        xu,xd,yu,yd=rect
        crop_img=img.crop([xu,yu,xd,yd])
        crop_img.show()

def test_do_ocr():
    image_path="tmp/images/1955-12/0.png"
    do_ocr(image_path)

test_do_ocr()
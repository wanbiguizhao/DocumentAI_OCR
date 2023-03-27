# 尝试使用easyocr识别图像。
import os
import easyocr 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(os.path.realpath( __file__))
)

reader=easyocr.Reader(["zh_tra"])

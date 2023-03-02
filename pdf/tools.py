
import os 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(__file__)
)
import sys 
sys.path.append(PROJECT_DIR)
from pdf.tempfilepath import TemporaryFilePath
from pdfminer.high_level import extract_text_to_fp
import re
from tqdm import tqdm
from tempfile import mktemp

def extract_images(input_file, output_dir):
    #output_dir = mkdtemp()
    with TemporaryFilePath() as output_file_name:
        print(output_file_name)
        with open(output_file_name,'wb' ) as tmp_file:
            with open(input_file,"rb") as pdff:
                extract_text_to_fp(pdff,outfp=tmp_file,output_dir=output_dir) 
                image_files=os.listdir(output_dir)
                print(image_files)

def batch_convert(pdfdir,image_save_dir):
    for pdfname in tqdm(os.listdir(pdfdir)):
        pdf_path=os.path.join(pdfdir,pdfname)
        image_dir=pdfname.split('.')[0]
        image_path=os.path.join(image_save_dir,image_dir)
        extract_images(pdf_path,image_path)
        print(image_path)
def batch_rename_img(base_image_dir):
    """
    把从PDF中提取的图片名称 I0.bmp-> 1954-01_00.bmp的形式
    """
    for image_dir_name in tqdm(os.listdir(base_image_dir)):
        image_dir_path=os.path.join(base_image_dir,image_dir_name)
        if not os.path.isdir(image_dir_path):
            continue
        for image_name in tqdm(os.listdir(image_dir_path)):
            pure_name,ext=os.path.splitext(image_name)
            if not  re.search('^I',pure_name):
                continue
            num=pure_name[1:]
            
            os.rename( 
                os.path.join(base_image_dir,image_dir_name,image_name),
                os.path.join(base_image_dir,image_dir_name,
                    "{}_{}{}".format(image_dir_name,num.rjust(2,'0'),ext)
                ),

            )
from PIL import Image 
def batch_convert_bmp_png(base_image_dir):
    for image_dir_name in tqdm(os.listdir(base_image_dir)):
        image_dir_path=os.path.join(base_image_dir,image_dir_name)
        if not os.path.isdir(image_dir_path):
            continue

        for image_name in tqdm(os.listdir(image_dir_path)):
            pure_name,ext=os.path.splitext(image_name)
            if ext=='.bmp' and not re.search('^\d+.*\d\d$',pure_name):
                continue
            Image.open( 
                os.path.join(base_image_dir,image_dir_name,image_name)
                ).save(
                    os.path.join(base_image_dir,image_dir_name,
                    "{}.png".format(pure_name)
                )
            )
            



if __name__=="__main__":
    #merge_labels_info(DATA_DIR,save_dir=os.path.join(PROJECT_DIR,'tmp'))
    in_f=os.path.join(PROJECT_DIR,'tmp','1954-01.pdf')
    out_dir=os.path.join(PROJECT_DIR,'tmp','1954-01')
    # extract_images(
    #     in_f,
    #     out_dir
    #     )
    base_image_dir=os.path.join(PROJECT_DIR,"tmp","images")
    #batch_convert("~/ocr/DocumentAI_OCR/tmp/pdfs","~/ocr/DocumentAI_OCR/tmp/images")
    #batch_rename_img( base_image_dir)
    batch_convert_bmp_png(base_image_dir)
    
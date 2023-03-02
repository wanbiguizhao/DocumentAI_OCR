
import os 
PROJECT_DIR= os.path.dirname(
    os.path.dirname(__file__)
)
import sys 
sys.path.append(PROJECT_DIR)
from pdf.tempfilepath import TemporaryFilePath
from pdfminer.high_level import extract_text_to_fp

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
    
if __name__=="__main__":
    #merge_labels_info(DATA_DIR,save_dir=os.path.join(PROJECT_DIR,'tmp'))
    in_f=os.path.join(PROJECT_DIR,'tmp','1954-01.pdf')
    out_dir=os.path.join(PROJECT_DIR,'tmp','1954-01')
    # extract_images(
    #     in_f,
    #     out_dir
    #     )

    batch_convert("/home/liukun/ocr/DocumentAI_OCR/tmp/pdfs","/home/liukun/ocr/DocumentAI_OCR/tmp/images")
    
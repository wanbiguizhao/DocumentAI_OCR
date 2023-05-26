# 对easyocr提供接口
# import os
# PROJECT_DIR= os.path.dirname(
#         os.path.dirname(os.path.realpath( __file__))
# )
from flask import Flask, render_template, request,jsonify
import json
import easyocr
import json
import numpy as np

app = Flask(__name__)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
reader=easyocr.Reader(["ch_tra"],gpu=True)
@app.route("/det",methods = ['GET', 'POST'])
def detect():
    if request.method =="POST":
      image_bytes = request.files['file'].read()
      result=reader.detect(image_bytes)
      print(result)
      return jsonify(json.loads(json.dumps(result,cls=NpEncoder)))
    html="""
    <html>
    <body>
      <form action = "/det" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>   
    </body>
    </html>
    """
    return html 
@app.route("/batch",methods = ['GET', 'POST'])
def batch_detect():
    if request.method =="POST":
      
      result_list=[]
      print(request.files)
      image_list=request.files.getlist("image")
      for image_data in image_list:
          #image_data[0] str name ;image_data[1] image byte ; 
          print(image_data.filename)
          image_bytes=image_data.read()
          result=reader.readtext(image_bytes)
          result_list.append(result)
      #result=reader.detect(image_bytes)
      print(len(image_list),len(result_list))
      return jsonify(json.loads(json.dumps(result_list,cls=NpEncoder)))
    html="""
    <html>
    <body>
      <form action = "/batch" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>   
    </body>
    </html>
    """
    return html
@app.route("/",methods = ['GET', 'POST'])
def hello_world():
    if request.method =="POST":
      image_bytes = request.files['file'].read()
      result=reader.readtext(image_bytes)
      print(result)
      return jsonify(json.loads(json.dumps(result,cls=NpEncoder)))
    html="""
    <html>
    <body>
      <form action = "/" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>   
    </body>
    </html>
    """
    return html
if __name__ == "__main__":
    app.run()
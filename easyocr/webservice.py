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
reader=easyocr.Reader(["ch_tra"],gpu=False)
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
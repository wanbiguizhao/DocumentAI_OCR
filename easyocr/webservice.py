# 对easyocr提供接口
import os
import io
from flask import Flask, render_template, request,jsonify
from werkzeug import security
import easyocr
import json
from PIL import Image 
import numpy as np
PROJECT_DIR= os.path.dirname(
        os.path.dirname(os.path.realpath( __file__))
    
)
app = Flask(__name__)
app.config['UPLOAD_FOLDER']=f"{PROJECT_DIR}/tmp/media"
app.config['MAX_CONTENT_PATH']=""
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
import json
reader=easyocr.Reader(["ch_tra"],gpu=False)
@app.route("/",methods = ['GET', 'POST'])
def hello_world():
    if request.method =="POST":
      f = request.files['file'].read()
    #   obj = io.BytesIO()
    #   while True:
    #     buf = f.stream.read(1024*16)
    #     if not buf:
    #         break
    #     obj(buf)
      
      #f.save(security.safe_join(app.config['UPLOAD_FOLDER'],f.filename))
      result=reader.readtext(f)
      print(result)
      
      return jsonify(json.loads( json.dumps(result,cls=NpEncoder)))
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
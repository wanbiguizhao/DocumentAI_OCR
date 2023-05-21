from flask import Flask, render_template, request,jsonify
import json

import paddle
import easyocr
import json
import numpy as np
from infer import load_model
from data.dataset import WIPByteDataset
cls_model=load_model()

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
def build_dataloader(image_byte,batch_size=256):
    from paddle.vision import transforms
    from paddle.io import DataLoader 
    from  mocov1.pp_infer import WIPDataset
    from mocov1.moco.loader import TwoCropsTransform
    from PIL import Image
    from mocov1.render import render_html
    normalize = transforms.Normalize(
            mean=[0.485], std=[0.229]
        )
        # 咱们就先弄mocov1的数据增强
    augmentation = [
            #transforms.RandomResizedCrop((16,48), scale=(0.2, 1.0)),
            #transforms.RandomGrayscale(p=0.2), 啥也别说了，paddle没有这个功能
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    wip=WIPByteDataset(image_byte,transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = DataLoader(
            wip,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            #pin_memory=True, paddle 没有过
            #sampler=None,
            drop_last=False,
        )
    return wip,train_loader
@app.route("/seg",methods = ['GET', 'POST'])
def hello_world():

    batch_size=256            
    if request.method =="POST":
        image_bytes = request.files['file'].read()
        wip,train_loader=build_dataloader(image_bytes,batch_size=batch_size)
        result=[]
        for k, (images, _) in enumerate(train_loader):    
            predict_info=cls_model(images[0])

            predict_labels=paddle.argmax(predict_info,axis=-1)# 预测的每个图片切片的类型。
            #predict_labels 根据这个重新图片的切割的位置。
            for index in range(len(predict_labels)):
                wip_index=index+k*batch_size
                seg_image_info = wip.data_list[wip_index]
                seg_beg_index=seg_image_info["seg_beg_index"]
                seg_end_index=seg_image_info["seg_end_index"]
                if int(predict_labels[index])==1:
                    mid_index=(seg_beg_index+seg_end_index)//2
                    result.append([wip_index,mid_index])
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
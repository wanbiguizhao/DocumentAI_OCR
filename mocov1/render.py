import os
PROJECT_DIR= os.path.dirname(
    os.path.dirname(os.path.realpath( __file__))
)
from jinja2 import Environment,FileSystemLoader, PackageLoader, select_autoescape
import re 
env = Environment(
    loader=FileSystemLoader(PROJECT_DIR),
    autoescape=select_autoescape()
)
template = env.get_template("mocov1/jinja/image_template.html")
image_info_table=[]
row_info=[]
col_count=0
print(os.getcwd())
for image_info in sorted(os.listdir("tmp/project_ocrSentences_dataset/word_image_slice")):
    if "png" not in image_info:
        continue
    if len(row_info)>40:
        image_info_table.append(row_info)
        row_info=[]
    
    matchObj = re.match(
        r'word.*_(?P<id_ds>\d+)_.*_(?P<id_cluster>\d+)', image_info)
    #print(matchObj)
    if not matchObj:
        continue
    id_ds=matchObj.groupdict()["id_ds"]
    id_cluster=matchObj.groupdict()["id_cluster"]
    row_info.append(
        {
            "path":image_info,
            "id_ds":id_ds[1:],
            "id_cluster":id_cluster
        }
    )
image_info_table.append(row_info)
#print(image_info_table)
html_data=template.render(image_info_table=image_info_table)
with open("tmp/project_ocrSentences_dataset/word_image_slice/imagetable.html",'w') as ith:
    ith.write(html_data)
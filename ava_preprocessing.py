from typing import List, Any
import pandas as pd
from detectron2.structures import BoxMode

from xmlr import xmlparse
from xmlr import xmliter
from xmlr import xmliter, XMLParsingMethods
import xml.etree.ElementTree
img_dir="/mnt/dst_datasets/own_omni_dataset/theodore_v3/images/"
import cv2
from matplotlib import pyplot as plt
count=0

df_cols = ["id","name",'xtl','ytl','xbr','ybr',"action_label","grp_id"]
rows=[]
for d in xmliter('/mnt/dst_datasets/own_omni_dataset/theodore_v3/theodore_plus_training.xml','image'):
    if count ==45000:
        record = {}
        boxes = []
        grp = []
        grp_id = []
        actions = []
        for k, v in d.items():
            if k == 'actions':
                for key, val in v.items():
                    for a in val:
                        actions.append(a["@name"])
                        grp_id.append(a["@group_id"])
            if k == 'box':
                if type(v) is list:
                    for x in v:
                        if x["@label"] == 'person':
                            b1 = []
                            b1.append(x['@xtl'])
                            b1.append(x['@ytl'])
                            b1.append(x['@xbr'])
                            b1.append(x['@ybr'])
                            boxes.append(b1)

            if k == '@id':
                id = v
            if k == '@width':
                w = v
            if k == '@name':
                n = v
            if k == "@height":
                h = v
        record["file_name"] = n
        record["image_id"] = id
        record["height"] = h
        record["width"] = w
        obj = {"bbox":boxes,"bbox_mode": BoxMode.XYXY_ABS}
        record['class_name'] = 'person'
        for i, j, k in zip(boxes, actions, grp_id):
            rows.append({"id": id, "name": n, "xtl": i[0], 'ytl': i[1], 'xbr': i[2], 'ybr': i[3], "action_label": j, "grp_id": k})
    count=count+1

out_df = pd.DataFrame(rows, columns=df_cols)
out_df.to_csv("/home/gnk/out_df.csv",index=False)










        






from detectron2.structures import BoxMode
import pandas as pd


#function to convert theodore datasets to coco format

def create_dataset_dicts(df1, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df1.name.unique()):
        record = {}
        image_df = df1[df1.name == img_name]
        file_path = img_name
        record["file_name"] = file_path
        record["image_id"] = image_id
        record["height"] = 1280
        record["width"] = 1280
        objs = []
        for _, row in image_df.iterrows():
            xmin = int(row.xtl)
            ymin = int(row.ytl)
            xmax = int(row.xbr)
            ymax = int(row.ybr)
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes.index(row.action_label),
                "iscrowd": 0
            }
            objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    new_data = []
    dataset = []
    for i in dataset_dicts:
        for key, value in i.items():
            if key == "file_name":
                if value not in new_data:
                    new_data.append(value)
                    dataset.append(i)
    return dataset

df=pd.read_csv("/home/gnk/verified_data/annotations/train_ver.csv")
print (df.head())
print(df['1'].value_counts())


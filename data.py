import pandas as pd
import numpy as np
from coco_type import *
import os
os.environ['DISPLAY'] = ':11.0'
import torch, torchvision
import os, json, cv2, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from detectron2.structures import BoxMode
df=pd.read_csv("/home/gnk/new/frame_lists/label3.csv")
df.name=df.name.replace({'/home/tschec/Downloads/theodore_plus':'/mnt/dst_datasets/own_omni_dataset/theodore_v3/images'}, regex=True)
df['action_label']="person"
df1=df.iloc[0:50]

#splitting  train and test
unique_files = df1.name.unique()
train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.95), replace=False))
train_df = df1[df1.name.isin(train_files)]
test_df = df1[~df1.name.isin(train_files)]
print (test_df)

#get class names
classes = df1.action_label.unique().tolist()
for d in ["train", "val"]:
  DatasetCatalog.register("person_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else test_df, classes))
  MetadataCatalog.get("person_" + d).set(thing_classes=classes)
statement_metadata = MetadataCatalog.get("person_train")
#register data in registery in the coco format

dataset_dicts = create_dataset_dicts(df1, classes)
for d in random.sample(dataset_dicts, 3):
    print (d['file_name'])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=statement_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    img=out.get_image()
    plt.imshow(img[:, :, ::-1])
    plt.show()

#eval helper
class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

#loading the config file
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
    "/home/gnk/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("/home/gnk/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

#datasets used for training and evaluation
cfg.DATASETS.TRAIN = ("faces_train",)
cfg.DATASETS.TEST = ("faces_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 100
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05


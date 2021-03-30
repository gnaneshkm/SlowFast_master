import cv2
import numpy as np
import pandas as pd
df=pd.read_csv("/home/gnk/new/frame_lists/val_new3_path.csv",sep=' ')
df1=df.iloc[0:15]
print (df1.tail())

df1.to_csv("/home/gnk/new/frame_lists/val_sample_path.csv",index=False,sep=" ")

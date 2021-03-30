import pandas as pd
df=pd.read_csv("/home/gnk/ava2/tools/detections_latest.csv")
df.columns =['video_name','video_id','xtl','ytl','xbr','ybr','label','score']
df.iloc[:,2:6]=df.iloc[:,2:6]*1024

df[['xtl','ytl','xbr','ybr']] = df[['xtl','ytl','xbr','ybr']].astype(int)
print (df.info())
print(df.head())
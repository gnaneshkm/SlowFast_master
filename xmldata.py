
import pandas as pd

#converter = xml2csv("/mnt/dst_datasets/own_omni_dataset/theodore_plus_v2/theodore_plus_training.xml", "/home/gnk/theodre.csv", encoding="utf-8")
#converter.convert(tag="item")
df = pd.read_csv('/home/gnk/out_df.csv')
x=list(df.id.unique())
y=x[0:1000]

def find_missing(lst):
    start = lst[0]
    end = lst[-1]
    return sorted(set(range(start, end + 1)).difference(lst))
z=find_missing(x)
print (z)
print (len(z))